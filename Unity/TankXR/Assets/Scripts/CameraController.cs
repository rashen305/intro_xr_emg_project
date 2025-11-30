using UnityEngine;

public class CameraController : MonoBehaviour
{
    public GameObject player;
    private Vector3 offset;
    
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        // Camera's relative position to the player. Compute initially because this won't change.
        offset = transform.position - player.transform.position;    
    }

    void LateUpdate()
    {
        // Update the camera's position to follow the player while maintaining the initial offset.
        transform.position = player.transform.position + offset; 
    }
}
