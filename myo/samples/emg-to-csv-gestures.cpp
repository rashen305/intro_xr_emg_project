// Copyright (C) 2013-2014 Thalmic Labs Inc.
// Distributed under the Myo SDK license agreement. See LICENSE.txt for details.

// EMG data collector with gesture-based CSV logging
// Collects 3-second samples with 2-second heads up, alternates between gesture and rest

#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <thread>
#include <vector>
#include <map>

#include <myo/myo.hpp>

// ============================================================================
// GESTURE DEFINITIONS
// ============================================================================
// Define your gestures here. Each gesture has a name and a label number.
// The label number will be written to the CSV file.
struct Gesture {
    std::string name;
    int label;
};

// Define your gestures here
const std::vector<Gesture> GESTURES = {
    {"clench", 1},   // TODO: Change "gesture1" to your gesture name
    {"spread", 2},   // TODO: Change "gesture2" to your gesture name
    {"flexion", 3},   // TODO: Change "gesture3" to your gesture name
    {"extension", 4},   // TODO: Change "gesture4" to your gesture name
    // Add more gestures as needed
    {"pinch", 5},
    // {"gesture6", 6},
};

// ============================================================================

class GestureDataCollector : public myo::DeviceListener {
public:
    enum class CycleState {
        HEADS_UP,      // 0-2s: Warning before collection
        COLLECTING     // 2-5s: Collecting data (3 seconds)
    };

    GestureDataCollector(const std::string& csvFilename, int selectedGestureIndex)
    : emgSamples()
    , csvFile(csvFilename)
    , sampleCount(0)
    , cycleStartTime(std::chrono::high_resolution_clock::now())
    , programStartTime(std::chrono::high_resolution_clock::now())
    , currentState(CycleState::HEADS_UP)
    , currentGestureIndex(selectedGestureIndex)
    , isCollectingGesture(true)  // Start with gesture
    , pairCount(0)
    {
        // Validate gesture index
        if (currentGestureIndex < 0 || currentGestureIndex >= static_cast<int>(GESTURES.size())) {
            throw std::runtime_error("Invalid gesture index selected");
        }
        
        // Write CSV header
        if (csvFile.is_open()) {
            csvFile << "timestamp,emg1,emg2,emg3,emg4,emg5,emg6,emg7,emg8,label\n";
            csvFile.flush();
        } else {
            throw std::runtime_error("Unable to open CSV file for writing");
        }
        
        // Print initial heads up (start with gesture)
        std::cout << "\n=== HEADS UP: Get ready for " << GESTURES[currentGestureIndex].name 
                  << " in 2 seconds... ===" << std::endl;
    }
    
    ~GestureDataCollector() {
        if (csvFile.is_open()) {
            csvFile.close();
        }
        std::cout << "\nTotal samples collected: " << sampleCount << std::endl;
        std::cout << "Total pairs completed: " << pairCount << std::endl;
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
        
        // Calculate time since cycle start
        auto currentTime = std::chrono::high_resolution_clock::now();
        double cycleElapsed = std::chrono::duration<double>(currentTime - cycleStartTime).count();
        double programElapsed = std::chrono::duration<double>(currentTime - programStartTime).count();
        
        // Update state based on cycle time
        // Cycle: 0-2s heads up, 2-5s collecting, then alternate
        CycleState newState = currentState;
        if (cycleElapsed < 2.0) {
            newState = CycleState::HEADS_UP;
        } else if (cycleElapsed < 5.0) {
            newState = CycleState::COLLECTING;
        } else {
            // Cycle complete, alternate between gesture and rest
            cycleStartTime = currentTime;
            isCollectingGesture = !isCollectingGesture;  // Alternate
            
            // Increment pair count when we complete a rest cycle (transitioning back to gesture)
            if (isCollectingGesture) {
                // We just finished rest, now starting gesture - increment pair count
                pairCount++;
            }
            
            newState = CycleState::HEADS_UP;
            std::string currentLabel = isCollectingGesture ? GESTURES[currentGestureIndex].name : "REST";
            std::cout << "\n=== HEADS UP: Get ready for " << currentLabel 
                      << " in 2 seconds... (Pair #" << pairCount << ") ===" << std::endl;
        }
        
        // Handle state transitions and notifications
        if (newState != currentState) {
            currentState = newState;
            
            if (currentState == CycleState::COLLECTING) {
                std::string currentLabel = isCollectingGesture ? GESTURES[currentGestureIndex].name : "REST";
                std::cout << "\n>>> COLLECTING " << currentLabel << " data... <<<" << std::endl;
            }
        }
        
        // Only write to CSV during collection phase
        if (currentState == CycleState::COLLECTING && csvFile.is_open()) {
            csvFile << std::fixed << std::setprecision(6) << programElapsed;
            for (int i = 0; i < 8; i++) {
                csvFile << "," << static_cast<int>(emg[i]);
            }
            // Use gesture label if collecting gesture, 0 for rest
            int label = isCollectingGesture ? GESTURES[currentGestureIndex].label : 0;
            csvFile << "," << label;
            csvFile << "\n";
            
            // Flush every 100 samples to ensure data is written
            if (sampleCount % 100 == 0) {
                csvFile.flush();
            }
            
            sampleCount++;
        }
    }

    // Print the current EMG values and state
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
        
        // Print current state and label
        std::string stateStr;
        switch (currentState) {
            case CycleState::HEADS_UP:
                stateStr = "HEADS UP";
                break;
            case CycleState::COLLECTING:
                stateStr = "COLLECTING";
                break;
        }
        std::string currentLabel = isCollectingGesture ? GESTURES[currentGestureIndex].name : "REST";
        std::cout << " [" << stateStr << " - " << currentLabel << "]";
        
        // Print cycle time and pair count
        auto currentTime = std::chrono::high_resolution_clock::now();
        double cycleElapsed = std::chrono::duration<double>(currentTime - cycleStartTime).count();
        std::cout << " Cycle: " << std::fixed << std::setprecision(1) << cycleElapsed << "s";
        std::cout << " | Pair #" << pairCount;

        std::cout << std::flush;
    }

    uint64_t getSampleCount() const { return sampleCount; }
    uint64_t getPairCount() const { return pairCount; }

private:
    std::array<int8_t, 8> emgSamples;
    std::ofstream csvFile;
    uint64_t sampleCount;
    std::chrono::high_resolution_clock::time_point cycleStartTime;
    std::chrono::high_resolution_clock::time_point programStartTime;
    CycleState currentState;
    int currentGestureIndex;  // Index of currently selected gesture
    bool isCollectingGesture; // true for gesture, false for rest
    uint64_t pairCount;       // Number of gesture/rest pairs completed
};

int main(int argc, char** argv)
{
    try {
        // Display gesture selection menu
        std::cout << "=====================================" << std::endl;
        std::cout << "EMG Gesture Data Collector" << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << "\nAvailable Gestures:" << std::endl;
        for (size_t i = 0; i < GESTURES.size(); i++) {
            std::cout << "  " << (i + 1) << ". " << GESTURES[i].name 
                      << " (label: " << GESTURES[i].label << ")" << std::endl;
        }
        std::cout << "\nSelect gesture to collect (1-" << GESTURES.size() << "): ";
        
        int selection;
        std::cin >> selection;
        
        if (selection < 1 || selection > static_cast<int>(GESTURES.size())) {
            throw std::runtime_error("Invalid gesture selection");
        }
        
        int selectedIndex = selection - 1;  // Convert to 0-based index
        std::cout << "\nSelected gesture: " << GESTURES[selectedIndex].name 
                  << " (label: " << GESTURES[selectedIndex].label << ")" << std::endl;
        
        // Generate filename with timestamp and gesture name
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::system_clock::to_time_t(now);
        std::stringstream filenameStream;
        filenameStream << "emg_data_" << GESTURES[selectedIndex].name << "_" << timestamp << ".csv";
        std::string filename = filenameStream.str();
        
        std::cout << "CSV file: " << filename << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << "Cycle: 2s heads up -> 3s collection (alternates gesture/rest)" << std::endl;
        std::cout << "Alternates between: " << GESTURES[selectedIndex].name << " (label: " 
                  << GESTURES[selectedIndex].label << ") and REST (label: 0)" << std::endl;
        std::cout << "=====================================" << std::endl;
        
        myo::Hub hub("com.example.emg-gesture-sample");

        std::cout << "\nAttempting to find a Myo..." << std::endl;

        myo::Myo* myo = hub.waitForMyo(10000);

        if (!myo) {
            throw std::runtime_error("Unable to find a Myo!");
        }

        std::cout << "Connected to a Myo armband!" << std::endl;

        // Enable EMG streaming at 200Hz
        myo->setStreamEmg(myo::Myo::streamEmgEnabled);

        // Create data collector with CSV file and selected gesture
        GestureDataCollector collector(filename, selectedIndex);

        hub.addListener(&collector);

        std::cout << "\nStarting collection cycles... Press Ctrl+C to stop" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

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

