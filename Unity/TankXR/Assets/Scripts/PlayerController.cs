using UnityEngine;
using UnityEngine.InputSystem;

[RequireComponent(typeof(Rigidbody))]
public class PlayerController : MonoBehaviour
{
    public float moveSpeed = 0.1f; // Translation speed in units per second (Units in Unity are typically meters)
    public float turnSpeed = 135.0f; // Degrees per second
    public float resetHeightY = 0.0f; // If the player falls below this Y value, reset position
    public float maxTiltAngle = 60.0f; // The angle (X or Z rotation) that defines "upright"
    public float resetTimeDuration = 5.0f; // The duration (in seconds) the object must be tilted to trigger reset

    // keyboard/controller input variables
    private float horizontalInput;
    private float verticalInput;
    // socket-based (EMG) input variables
    private float socketHorizontalInput;
    private float socketVerticalInput;

    private Rigidbody rb;
    private Vector3 startPosition;
    private Quaternion startRotation;

    // *** NEW STATE VARIABLES ***
    private float timeUpsideDown = 0.0f; // Tracks how long the object has been tilted past the maxTiltAngle
    
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        rb = GetComponent<Rigidbody>();
        
        // Ensure Rigidbody exists and is set up for rotation/translation control
        if (rb == null)
        {
            Debug.LogError("Rigidbody component not found on the PlayerController object.");
            enabled = false; // Disable the script if we can't control physics
            return;
        }

        startPosition = transform.position;
        startRotation = transform.rotation;
    }

    // Update is called once per frame
    void Update()
    {
       // 1. Capture Input (Do this in Update, not FixedUpdate)
        horizontalInput = Input.GetAxis("Horizontal"); // For turning
        verticalInput = Input.GetAxis("Vertical");   // For forward/backward 

        // 2. Check for the Reset Condition (Falling below Y height)
        if (transform.position.y < resetHeightY)
        {
            ResetPlayerPosition();
        }

        // *** NEW: Handle Upside-Down Timer ***
        // 1. Check if the object is tilted past the defined limit.
        if (IsTilted())
        {
            // If tilted, start accumulating time. Time.deltaTime is used for frame-rate independent timing.
            timeUpsideDown += Time.deltaTime;

            // 2. Check if the accumulated time exceeds the limit.
            if (timeUpsideDown >= resetTimeDuration)
            {
                Debug.Log($"Resetting player: Tilted for {timeUpsideDown} seconds.");
                ResetPlayerPosition();
            }
        }
        else
        {
            // If the object is upright, reset the timer.
            timeUpsideDown = 0.0f;
        }
    }

    // New Function: Checks if the object is severely tilted (near upside down).
    bool IsTilted()
    {
        // Get the current rotation's Euler angles
        Vector3 currentRotation = transform.rotation.eulerAngles;

        // Normalize angles to be between -180 and 180 for reliable comparison against maxTiltAngle
        float angleX = NormalizeAngle(currentRotation.x);
        float angleZ = NormalizeAngle(currentRotation.z);

        // Check if the absolute rotation around the X or Z axis exceeds the threshold
        if (Mathf.Abs(angleX) > maxTiltAngle || Mathf.Abs(angleZ) > maxTiltAngle)
        {
            return true;
        }
        return false;
    }
    
    // Helper function to normalize an angle from (0, 360) to (-180, 180)
    float NormalizeAngle(float angle)
    {
        if (angle > 180f)
        {
            return angle - 360f;
        }
        return angle;
    }

    // 3. The Reset Function
    void ResetPlayerPosition()
    {
        // Reset the position to the stored start position
        transform.SetPositionAndRotation(startPosition, startRotation);

        // Stop all movement and rotation (Crucial for Rigidbody objects)
        if (rb != null)
        {
            rb.linearVelocity = Vector3.zero; // Note: Use .velocity, not .linearVelocity for standard Rigidbody
            rb.angularVelocity = Vector3.zero;
        }
        
        // Reset the timer immediately upon successful reset
        timeUpsideDown = 0.0f; 
    }

    // FixedUpdate is called at a fixed interval and is independent of frame rate. 
    // Good for physics updates.
    private void FixedUpdate()
    {
        // --- COMBINE KEYBOARD AND SOCKET INPUT ---
        // Combine inputs. The design here assumes keyboard input overrides or adds to socket input.
        // For simplicity, we'll let them add up, or you can choose one source to be dominant.
        float finalHorizontalInput = horizontalInput + socketHorizontalInput;
        float finalVerticalInput = verticalInput + socketVerticalInput;
        
        // 2. Handle Rotation (Left/Right)
        float turn = finalHorizontalInput * turnSpeed * Time.fixedDeltaTime;
        Quaternion turnRotation = Quaternion.Euler(0f, turn, 0f);
        rb.MoveRotation(rb.rotation * turnRotation);

        // 3. Handle Translation (Forward/Backward)
        Vector3 movement = moveSpeed * Time.fixedDeltaTime * finalVerticalInput * transform.forward;
        rb.MovePosition(rb.position + movement);
        
        // Optional: Stop the object from spinning wildly due to physics
        // TODO: remove? or just cap maximum angular velocity
        rb.angularVelocity = Vector3.zero;
    }

    // ====================================================================
    // External Method for SocketReceiver to control the player
    // ====================================================================
    // Processes a DataPacket received from the SocketReceiver and applies the action to the player.
    // This runs on the main Unity thread (called from SocketReceiver's Update).
    public void ApplySocketInput(DataPacket packet)
    {
        // Clear any previous input impulse from a potentially fast Update loop
        // If a new packet arrives, it overrides the previous one for the next FixedUpdate cycle.
        socketHorizontalInput = 0.0f;
        socketVerticalInput = 0.0f;
        
        switch (packet.classification)
        {
            case "rest":
                // No movement
                break;
            case "clench":
                // Set the virtual vertical input to move forward
                // Using 1.0f mimics a full key press (like Input.GetAxis, which returns -1 to 1)
                socketVerticalInput = 1.0f; 
                Debug.Log($"[SOCKET INPUT] Setting Vertical Input: Forward");
                break;
            case "spread":
                socketVerticalInput = -1.0f;
                Debug.Log("[SOCKET INPUT] Setting Vertical Input: Backward");
                break;
            case "flexion":
                // Set the virtual horizontal input to turn left
                socketHorizontalInput = -1.0f;
                Debug.Log($"[SOCKET INPUT] Setting Horizontal Input: Left");
                break;
            case "extension":
                // Set the virtual horizontal input to turn right
                socketHorizontalInput = 1.0f;
                Debug.Log($"[SOCKET INPUT] Setting Horizontal Input: Right");
                break;
            default:
                Debug.LogWarning($"[SOCKET INPUT] Unknown classification: {packet.classification}");
                break;
        }
    }
}