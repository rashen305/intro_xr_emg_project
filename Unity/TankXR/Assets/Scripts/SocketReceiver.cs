using UnityEngine;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Collections.Generic;
using System.IO;

// --- Data Structure for JSON Payload ---
// This class must match the structure of the JSON sent by the Python script:
// {"classification": "clench", "force": 0.5}
[Serializable]
public class DataPacket
{
    public string classification;
    public float force;

    public override string ToString()
    {
        return $"[Classification: {classification}, Force: {force}]";
    }
}

[RequireComponent(typeof(PlayerController))]
public class SocketReceiver : MonoBehaviour
{
    // --- Configuration ---
    private const string HOST = "0.0.0.0"; // Listen on all available network interfaces
    private const int TCP_PORT = 9000;
    private const int UDP_PORT = 9001;
    
    // --- TCP Variables ---
    private Thread tcpReceiveThread;
    private TcpListener tcpListener;
    private TcpClient tcpClient;
    // Buffer to hold partial TCP data if a packet is split across reads
    private string tcpRemainingData = string.Empty;


    // --- UDP Variables ---
    private Thread udpReceiveThread;
    private UdpClient udpClient;

    // --- Reference to PlayerController ---
    private PlayerController playerController;

    // --- Thread-Safe Data Queue ---
    // Use a queue to safely transfer received data from the background threads 
    // to the main Unity thread (Update/FixedUpdate)
    private readonly Queue<DataPacket> dataQueue = new();
    private readonly object queueLock = new();

    private void Start()
    {
        // 1. Find the PlayerController on this same GameObject
        playerController = GetComponent<PlayerController>();
        if (playerController == null)
        {
            Debug.LogError("PlayerController script not found on this GameObject. SocketReceiver requires it.");
            enabled = false;
            return;
        }

        Debug.Log("Starting Socket Receiver...");        
        // Start both listeners
        StartTCPListener();
        StartUDPListener();
    }

    // ====================================================================
    // 1. TCP Listener Setup (Connection-based, reliable)
    // ====================================================================

    private void StartTCPListener()
    {
        try
        {
            tcpListener = new TcpListener(IPAddress.Parse(HOST), TCP_PORT);
            // Allow the socket to be reused immediately after closure, preventing port conflicts during quick restarts
            tcpListener.Server.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
            tcpListener.Start();

            // Start the TCP listener thread
            tcpReceiveThread = new Thread(new ThreadStart(TCPListenerLoop));
            tcpReceiveThread.IsBackground = true;
            tcpReceiveThread.Start();
            Debug.Log($"TCP Listener started on {HOST}:{TCP_PORT}. Waiting for persistent connection...");
        }
        catch (Exception e)
        {
            Debug.LogError($"TCP Error starting listener: {e.Message}");
        }
    }

    private void TCPListenerLoop()
    {
        try
        {
            // IMPORTANT: The Python sender now opens ONE persistent connection, 
            // so we wait for it once and then stay in the reading loop.
            
            // Block until the single client connects (the Python script)
            tcpClient = tcpListener.AcceptTcpClient();
            Debug.Log("TCP Client connected!");
            
            // Set client receive timeout to prevent thread from blocking indefinitely 
            // if data flow stops, allowing disconnect check.
            tcpClient.ReceiveTimeout = 1000; 

            NetworkStream stream = tcpClient.GetStream();
            byte[] buffer = new byte[1024];
            int bytesRead;

            // Inner loop to continuously read data from the connected client stream
            while (tcpClient.Connected)
            {
                try 
                {
                    // Read data from the stream (this is blocking but with a timeout)
                    bytesRead = stream.Read(buffer, 0, buffer.Length);
                    
                    if (bytesRead > 0)
                    {
                        string dataChunk = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                        
                        // Prepend any remaining data from the previous read
                        string fullStream = tcpRemainingData + dataChunk;
                        
                        // Split the stream by the newline delimiter
                        string[] packets = fullStream.Split(new char[] { '\n' }, StringSplitOptions.None);

                        // All but the last element are complete packets
                        for (int i = 0; i < packets.Length - 1; i++)
                        {
                            if (!string.IsNullOrEmpty(packets[i]))
                            {
                                ParseAndQueueData(packets[i], "TCP");
                            }
                        }
                        
                        // The last element is either a complete packet (if the stream ended in '\n') 
                        // or a partial packet. We save it for the next read.
                        tcpRemainingData = packets[packets.Length - 1];
                    }
                    else if (bytesRead == 0)
                    {
                        // bytesRead == 0 indicates a graceful disconnect by the client (Python)
                        Debug.Log("TCP Client performed a graceful disconnect (bytesRead == 0).");
                        break; 
                    }
                }
                catch (IOException ioEx) when (ioEx.InnerException is SocketException sockEx && sockEx.SocketErrorCode == SocketError.TimedOut)
                {
                    // Ignore read timeouts; this just means no data arrived in 1 second.
                }
                catch (Exception readEx)
                {
                    Debug.LogError($"TCP Stream Read Error: {readEx.Message}");
                    break;
                }
                
                // Small pause to yield control if the stream is idle
                Thread.Sleep(1);
            }
        }
        catch (SocketException sockEx) when (sockEx.SocketErrorCode == SocketError.Interrupted || sockEx.SocketErrorCode == SocketError.OperationAborted)
        {
            // Expected when the listener is stopped during AcceptTcpClient
        }
        catch (ThreadAbortException)
        {
            // Expected when the thread is aborted during cleanup
        }
        catch (Exception e)
        {
            Debug.LogError($"TCP Listener Loop Error: {e.Message}");
        }
        finally
        {
            // If the connection breaks, attempt to clean up and let Python reconnect.
            if (tcpClient != null)
            {
                tcpClient.Close();
                tcpClient = null;
            }
            // For persistence, we restart the whole listener to accept the Python reconnection
            if (tcpListener != null)
            {
                Debug.Log("TCP connection ended. Restarting TCP listener to wait for Python reconnection.");
                tcpListener.Stop();
                tcpListener = null;
                StartTCPListener();
            }
        }
    }


    // ====================================================================
    // 2. UDP Listener Setup (Datagram-based, unreliable)
    // ====================================================================

    private void StartUDPListener()
    {
        try
        {
            udpClient = new UdpClient(UDP_PORT);
            // Allow the socket to be reused immediately after closure
            udpClient.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);


            // Start the UDP listener thread
            udpReceiveThread = new Thread(new ThreadStart(UDPListenerLoop));
            udpReceiveThread.IsBackground = true;
            udpReceiveThread.Start();
            Debug.Log($"UDP Listener started on {HOST}:{UDP_PORT}. Ready for datagrams...");
        }
        catch (Exception e)
        {
            Debug.LogError($"UDP Error starting listener: {e.Message}");
        }
    }

    private void UDPListenerLoop()
    {
        try
        {
            // IPEndPoint object will allow us to read incoming data 
            // from any IP address/port combo
            IPEndPoint remoteIp = new IPEndPoint(IPAddress.Any, 0);

            while (true)
            {
                // Receive method blocks until a datagram is received
                byte[] data = udpClient.Receive(ref remoteIp);
                string message = Encoding.UTF8.GetString(data);
                
                ParseAndQueueData(message, "UDP");
            }
        }
        catch (SocketException sockEx) when (sockEx.SocketErrorCode == SocketError.Interrupted || sockEx.SocketErrorCode == SocketError.ConnectionReset)
        {
            // Expected when the thread is aborted or if the connection reset happens (less common in UDP)
        }
        catch (ThreadAbortException)
        {
            // Expected when the thread is aborted during cleanup
        }
        catch (Exception e)
        {
            Debug.LogError($"UDP Listener Loop Error: {e.Message}");
        }
    }

    // ====================================================================
    // 3. Data Processing and Main Thread Update
    // ====================================================================

    private void ParseAndQueueData(string jsonString, string protocol)
    {
        try
        {
            // Parse the JSON string into the C# DataPacket object
            DataPacket receivedPacket = JsonUtility.FromJson<DataPacket>(jsonString);
            
            // Add to the thread-safe queue
            lock (queueLock)
            {
                dataQueue.Enqueue(receivedPacket);
            }

            Debug.Log($"[Received via {protocol}] Successfully parsed packet: {receivedPacket.ToString()}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to parse JSON string: '{jsonString}' - Error: {e.Message}");
        }
    }

    // Update is called once per frame on the main thread
    private void Update()
    {
        // Check the queue on the main thread for new data
        lock (queueLock)
        {
            while (dataQueue.Count > 0)
            {
                DataPacket packet = dataQueue.Dequeue();
                
                // --- ACTION: DECOUPLED CALL TO PLAYERCONTROLLER! ---
                if (playerController != null)
                {
                    playerController.ApplySocketInput(packet);
                }
            }
        }
    }

    // ====================================================================
    // 4. Cleanup on Application Quit
    // ====================================================================
    private void OnDisable()
    {
        // Clean up TCP resources
        if (tcpListener != null)
        {
            tcpListener.Stop();
            tcpListener = null; 
            Debug.Log("TCP Listener stopped.");
        }
        if (tcpClient != null)
        {
            tcpClient.Close();
        }
        if (tcpReceiveThread != null && tcpReceiveThread.IsAlive)
        {
            tcpReceiveThread.Interrupt(); 
            // tcpReceiveThread.Abort(); // Abort is less preferred, relying on Interrupt/Stop
        }

        // Clean up UDP resources
        if (udpClient != null)
        {
            udpClient.Close();
            Debug.Log("UDP Listener stopped.");
        }
        if (udpReceiveThread != null && udpReceiveThread.IsAlive)
        {
            udpReceiveThread.Abort();
        }
        
        Debug.Log("Socket Receiver cleanup complete.");
    }
}