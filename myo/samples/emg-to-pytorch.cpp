// Copyright (C) 2013-2014 Thalmic Labs Inc.
// Distributed under the Myo SDK license agreement. See LICENSE.txt for details.

// EMG data collector with TCP socket transmission to PyTorch model
// Compile with: g++ emg-to-pytorch.cpp -I../include -L../lib -lmyo64 -lws2_32 -o emg-to-pytorch

#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <thread>
#include <mutex>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

#include <myo/myo.hpp>

class SocketSender {
public:
    SocketSender(const std::string& host, int port) : host_(host), port_(port), socket_(-1), connected_(false) {
#ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            throw std::runtime_error("WSAStartup failed");
        }
#endif
    }
    
    ~SocketSender() {
        disconnect();
#ifdef _WIN32
        WSACleanup();
#endif
    }
    
    bool connect() {
        if (connected_) return true;
        
#ifdef _WIN32
        socket_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (socket_ == INVALID_SOCKET) {
            std::cerr << "Socket creation failed" << std::endl;
            return false;
        }
        
        sockaddr_in serverAddr;
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(port_);
        inet_pton(AF_INET, host_.c_str(), &serverAddr.sin_addr);
        
        if (::connect(socket_, (sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
            std::cerr << "Connection failed to " << host_ << ":" << port_ << std::endl;
            closesocket(socket_);
            socket_ = -1;
            return false;
        }
#else
        socket_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (socket_ < 0) {
            std::cerr << "Socket creation failed" << std::endl;
            return false;
        }
        
        sockaddr_in serverAddr;
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(port_);
        inet_pton(AF_INET, host_.c_str(), &serverAddr.sin_addr);
        
        if (::connect(socket_, (sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
            std::cerr << "Connection failed" << std::endl;
            close(socket_);
            socket_ = -1;
            return false;
        }
#endif
        
        connected_ = true;
        std::cout << "Connected to PyTorch receiver at " << host_ << ":" << port_ << std::endl;
        return true;
    }
    
    void disconnect() {
        if (socket_ >= 0) {
#ifdef _WIN32
            closesocket(socket_);
#else
            close(socket_);
#endif
            socket_ = -1;
            connected_ = false;
        }
    }
    
    bool sendData(const std::string& data) {
        if (!connected_ && !connect()) {
            return false;
        }
        
        std::string message = data + "\n";
        int bytesSent;
        
#ifdef _WIN32
        bytesSent = ::send(socket_, message.c_str(), message.length(), 0);
        if (bytesSent == SOCKET_ERROR) {
            connected_ = false;
            disconnect();
            return false;
        }
#else
        bytesSent = ::send(socket_, message.c_str(), message.length(), 0);
        if (bytesSent < 0) {
            connected_ = false;
            disconnect();
            return false;
        }
#endif
        
        return true;
    }
    
    bool isConnected() const { return connected_; }
    
private:
    std::string host_;
    int port_;
#ifdef _WIN32
    SOCKET socket_;
#else
    int socket_;
#endif
    bool connected_;
};

class DataCollector : public myo::DeviceListener {
public:
    DataCollector(const std::string& csvFilename, SocketSender* socketSender = nullptr)
    : emgSamples()
    , csvFile(csvFilename)
    , socketSender_(socketSender)
    , sampleCount(0)
    , startTime(std::chrono::high_resolution_clock::now())
    {
        // Write CSV header
        if (csvFile.is_open()) {
            csvFile << "timestamp,sample_number,emg1,emg2,emg3,emg4,emg5,emg6,emg7,emg8\n";
            csvFile.flush();
        } else {
            throw std::runtime_error("Unable to open CSV file for writing");
        }
    }
    
    ~DataCollector() {
        if (csvFile.is_open()) {
            csvFile.close();
        }
        std::cout << "\nTotal samples collected: " << sampleCount << std::endl;
    }

    void onUnpair(myo::Myo* myo, uint64_t timestamp)
    {
        emgSamples.fill(0);
    }

    void onEmgData(myo::Myo* myo, uint64_t timestamp, const int8_t* emg)
    {
        // Update EMG samples
        for (int i = 0; i < 8; i++) {
            emgSamples[i] = emg[i];
        }
        
        // Calculate elapsed time in seconds
        auto currentTime = std::chrono::high_resolution_clock::now();
        double elapsedTime = std::chrono::duration<double>(currentTime - startTime).count();
        
        // Write to CSV
        if (csvFile.is_open()) {
            csvFile << std::fixed << std::setprecision(6) << elapsedTime << ","
                    << sampleCount;
            for (int i = 0; i < 8; i++) {
                csvFile << "," << static_cast<int>(emg[i]);
            }
            csvFile << "\n";
            
            // Flush every 100 samples to ensure data is written
            if (sampleCount % 100 == 0) {
                csvFile.flush();
            }
        }
        
        // Send to PyTorch via socket (JSON format)
        if (socketSender_ && socketSender_->isConnected()) {
            std::ostringstream jsonStream;
            jsonStream << "{"
                       << "\"timestamp\":" << std::fixed << std::setprecision(6) << elapsedTime << ","
                       << "\"sample\":" << sampleCount << ","
                       << "\"emg\":[";
            for (int i = 0; i < 8; i++) {
                if (i > 0) jsonStream << ",";
                jsonStream << static_cast<int>(emg[i]);
            }
            jsonStream << "]}";
            
            socketSender_->sendData(jsonStream.str());
        }
        
        sampleCount++;
    }

    // Print the current EMG values
    void print()
    {
        // Clear the current line
        std::cout << '\r';

        // Print out the EMG data
        for (size_t i = 0; i < emgSamples.size(); i++) {
            std::ostringstream oss;
            oss << static_cast<int>(emgSamples[i]);
            std::string emgString = oss.str();

            std::cout << '[' << emgString << std::string(4 - emgString.size(), ' ') << ']';
        }
        
        // Show connection status
        if (socketSender_) {
            std::cout << (socketSender_->isConnected() ? " [TCP:OK]" : " [TCP:--]");
        }

        std::cout << std::flush;
    }

    uint64_t getSampleCount() const { return sampleCount; }

private:
    std::array<int8_t, 8> emgSamples;
    std::ofstream csvFile;
    SocketSender* socketSender_;
    uint64_t sampleCount;
    std::chrono::high_resolution_clock::time_point startTime;
};

int main(int argc, char** argv)
{
    // Default socket settings
    std::string socketHost = "127.0.0.1";
    int socketPort = 9002;
    
    // Parse command line arguments
    bool enableSocket = true;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--no-socket") {
            enableSocket = false;
        } else if (arg == "--host" && i + 1 < argc) {
            socketHost = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            socketPort = std::stoi(argv[++i]);
        }
    }
    
    try {
        // Generate filename with timestamp
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::system_clock::to_time_t(now);
        std::stringstream filenameStream;
        filenameStream << "emg_data_" << timestamp << ".csv";
        std::string filename = filenameStream.str();
        
        std::cout << "=====================================" << std::endl;
        std::cout << "EMG Data Collector - 200Hz Sampling" << std::endl;
        std::cout << "CSV file: " << filename << std::endl;
        if (enableSocket) {
            std::cout << "Socket: " << socketHost << ":" << socketPort << std::endl;
        }
        std::cout << "=====================================" << std::endl;
        
        // Create socket sender
        SocketSender* socketSender = nullptr;
        if (enableSocket) {
            socketSender = new SocketSender(socketHost, socketPort);
            // Try to connect (will retry on first send if fails)
            socketSender->connect();
        }
        
        myo::Hub hub("com.example.emg-data-sample");

        std::cout << "Attempting to find a Myo..." << std::endl;

        myo::Myo* myo = hub.waitForMyo(10000);

        if (!myo) {
            throw std::runtime_error("Unable to find a Myo!");
        }

        std::cout << "Connected to a Myo armband!" << std::endl;

        // Enable EMG streaming at 200Hz
        myo->setStreamEmg(myo::Myo::streamEmgEnabled);

        // Create data collector with CSV file and socket sender
        DataCollector collector(filename, socketSender);

        hub.addListener(&collector);

        std::cout << "Recording... Press Ctrl+C to stop" << std::endl;

        // Main loop - run at 200Hz (5ms intervals)
        while (1) {
            hub.run(5);
            // Print the current EMG values after processing events
            collector.print();
        }

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        std::cerr << "Press enter to continue.";
        std::cin.ignore();
        return 1;
    }
}

