// Copyright (C) 2013-2014 Thalmic Labs Inc.
// Distributed under the Myo SDK license agreement. See LICENSE.txt for details.

// EMG and Force data collector
// Collects EMG data at 200Hz and Phidget load cell data at 100Hz

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

#include <myo/myo.hpp>
#include "phidget22.h"

class ForceEMGCollector : public myo::DeviceListener {
public:
    enum CollectionState {
        WAITING,
        COLLECTING
    };

    ForceEMGCollector(const std::string& csvFilename, PhidgetVoltageRatioInputHandle phidgetHandle)
        : emgSamples()
        , csvFile(csvFilename)
        , sampleCount(0)
        , programStartTime(std::chrono::high_resolution_clock::now())
        , phidgetCh(phidgetHandle)
        , currentWeight(0.0)
        , dataMutex()
        , state(WAITING)
        , stateStartTime(std::chrono::high_resolution_clock::now())
        , waitDuration(10.0)
        , collectDuration(10.0)
    {
        // Calibration constants
        gain = -4839.02337806184;
        offset = -1.3201e-5;
        
        // Write CSV header
        if (csvFile.is_open()) {
            csvFile << "timestamp,emg1,emg2,emg3,emg4,emg5,emg6,emg7,emg8,label\n";
            csvFile.flush();
        } else {
            throw std::runtime_error("Unable to open CSV file for writing");
        }
        
        std::cout << "Force and EMG collector initialized" << std::endl;
        std::cout << "EMG rate: 200Hz, Phidget rate: 100Hz" << std::endl;
        std::cout << "Gain: " << gain << ", Offset: " << offset << std::endl;
        std::cout << "\n=== TIMING PATTERN ===" << std::endl;
        std::cout << "10 seconds WAITING (showing weight)" << std::endl;
        std::cout << "10 seconds COLLECTING (recording data)" << std::endl;
        std::cout << "10 seconds WAITING..." << std::endl;
        std::cout << "=======================\n" << std::endl;
    }
    
    ~ForceEMGCollector() {
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
        
        // Get current weight from Phidget
        double voltageRatio = 0.0;
        PhidgetReturnCode res = PhidgetVoltageRatioInput_getVoltageRatio(phidgetCh, &voltageRatio);
        
        double weight = 0.0;
        if (res == EPHIDGET_OK) {
            // Convert voltage ratio to weight in kg
            weight = (voltageRatio * gain) + offset;
            
            std::lock_guard<std::mutex> lock(dataMutex);
            currentWeight = weight;
        } else {
            // Use last known weight if read fails
            std::lock_guard<std::mutex> lock(dataMutex);
            weight = currentWeight;
        }
        
        // Calculate timestamp
        auto currentTime = std::chrono::high_resolution_clock::now();
        double programElapsed = std::chrono::duration<double>(currentTime - programStartTime).count();
        
        // Only write to CSV when COLLECTING
        if (state == COLLECTING && csvFile.is_open()) {
            csvFile << std::fixed << std::setprecision(6) << programElapsed;
            for (int i = 0; i < 8; i++) {
                csvFile << "," << static_cast<int>(emg[i]);
            }
            csvFile << "," << std::setprecision(9) << weight;
            csvFile << "\n";
            
            // Flush every 100 samples to ensure data is written
            if (sampleCount % 100 == 0) {
                csvFile.flush();
            }
            
            sampleCount++;
        }
    }

    // Update state based on timing
    void updateState()
    {
        auto currentTime = std::chrono::high_resolution_clock::now();
        double stateElapsed = std::chrono::duration<double>(currentTime - stateStartTime).count();
        
        if (state == WAITING && stateElapsed >= waitDuration) {
            // Switch to COLLECTING
            state = COLLECTING;
            stateStartTime = currentTime;
            std::cout << "\n\n*** STARTING DATA COLLECTION ***\n" << std::endl;
        } else if (state == COLLECTING && stateElapsed >= collectDuration) {
            // Switch to WAITING
            state = WAITING;
            stateStartTime = currentTime;
            csvFile.flush(); // Ensure all data is written
            std::cout << "\n\n*** PAUSING - Get ready for next collection ***\n" << std::endl;
        }
    }

    // Print the current EMG and weight values
    void print()
    {
        // Clear the current line
        std::cout << '\r';
        
        // Calculate time remaining in current state
        auto currentTime = std::chrono::high_resolution_clock::now();
        double stateElapsed = std::chrono::duration<double>(currentTime - stateStartTime).count();
        double timeRemaining = 0.0;
        
        if (state == WAITING) {
            timeRemaining = waitDuration - stateElapsed;
            std::cout << "[WAITING " << std::fixed << std::setprecision(1) << timeRemaining << "s] ";
        } else {
            timeRemaining = collectDuration - stateElapsed;
            std::cout << "[COLLECTING " << std::fixed << std::setprecision(1) << timeRemaining << "s] ";
        }

        // Print current weight (always visible)
        {
            std::lock_guard<std::mutex> lock(dataMutex);
            std::cout << "[Weight: " << std::fixed << std::setprecision(3) << currentWeight << " kg] ";
        }
        
        // Print out the EMG data
        for (size_t i = 0; i < emgSamples.size(); i++) {
            std::ostringstream oss;
            oss << static_cast<int>(emgSamples[i]);
            std::string emgString = oss.str();

            std::cout << '[' << emgString << std::string(4 - emgString.size(), ' ') << ']';
        }
        
        std::cout << " [Samples: " << sampleCount << "]";

        std::cout << std::flush;
    }

    uint64_t getSampleCount() const { return sampleCount; }

private:
    std::array<int8_t, 8> emgSamples;
    std::ofstream csvFile;
    uint64_t sampleCount;
    std::chrono::high_resolution_clock::time_point programStartTime;
    PhidgetVoltageRatioInputHandle phidgetCh;
    double currentWeight;
    double gain;
    double offset;
    std::mutex dataMutex;
    
    // Timing control
    CollectionState state;
    std::chrono::high_resolution_clock::time_point stateStartTime;
    double waitDuration;     // seconds
    double collectDuration;  // seconds
};

int main(int argc, char** argv)
{
    PhidgetVoltageRatioInputHandle phidgetCh = NULL;
    
    try {
        // Generate filename with timestamp
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::system_clock::to_time_t(now);
        std::stringstream filenameStream;
        filenameStream << "force_and_emg_data_" << timestamp << ".csv";
        std::string filename = filenameStream.str();
        
        std::cout << "=====================================" << std::endl;
        std::cout << "Force and EMG Data Collector" << std::endl;
        std::cout << "CSV file: " << filename << std::endl;
        std::cout << "=====================================" << std::endl;
        
        // Initialize Phidget
        std::cout << "Initializing Phidget load cell..." << std::endl;
        PhidgetReturnCode res;
        
        res = PhidgetVoltageRatioInput_create(&phidgetCh);
        if (res != EPHIDGET_OK) {
            throw std::runtime_error("Failed to create Phidget VoltageRatioInput");
        }
        
        // Set data interval to 10ms (100Hz)
        res = Phidget_setDataInterval((PhidgetHandle)phidgetCh, 10);
        if (res != EPHIDGET_OK) {
            std::cerr << "Warning: Could not set data interval, will use default" << std::endl;
        }
        
        // Enable bridge if needed (for load cells)
        res = PhidgetVoltageRatioInput_setBridgeEnabled(phidgetCh, 1);
        if (res != EPHIDGET_OK) {
            std::cerr << "Warning: Could not enable bridge, continuing anyway" << std::endl;
        }
        
        // Open and wait for attachment
        res = Phidget_openWaitForAttachment((PhidgetHandle)phidgetCh, 5000);
        if (res != EPHIDGET_OK) {
            throw std::runtime_error("Failed to open Phidget or timeout waiting for attachment");
        }
        
        std::cout << "Phidget load cell connected!" << std::endl;
        
        // Initialize Myo
        std::cout << "Attempting to find a Myo..." << std::endl;
        myo::Hub hub("com.example.force-emg-sample");

        myo::Myo* myo = hub.waitForMyo(10000);

        if (!myo) {
            throw std::runtime_error("Unable to find a Myo!");
        }

        std::cout << "Connected to a Myo armband!" << std::endl;

        // Enable EMG streaming at 200Hz
        myo->setStreamEmg(myo::Myo::streamEmgEnabled);

        // Create data collector with CSV file
        ForceEMGCollector collector(filename, phidgetCh);

        hub.addListener(&collector);

        std::cout << "\nStarting data collection... Press Ctrl+C to stop" << std::endl;
        std::cout << "First 10 seconds: WAITING (get your weight ready)" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // Main loop - run at 200Hz (5ms intervals)
        while (1) {
            hub.run(5);
            // Update state based on timing
            collector.updateState();
            // Print the current EMG and weight values after processing events
            collector.print();
        }

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        
        // Clean up Phidget
        if (phidgetCh != NULL) {
            Phidget_close((PhidgetHandle)phidgetCh);
            PhidgetVoltageRatioInput_delete(&phidgetCh);
        }
        
        std::cerr << "Press enter to continue.";
        std::cin.ignore();
        return 1;
    }
    
    // Clean up Phidget (won't reach here normally due to Ctrl+C exit)
    if (phidgetCh != NULL) {
        Phidget_close((PhidgetHandle)phidgetCh);
        PhidgetVoltageRatioInput_delete(&phidgetCh);
    }
    
    return 0;
}

