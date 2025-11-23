// Copyright (C) 2013-2014 Thalmic Labs Inc.
// Distributed under the Myo SDK license agreement. See LICENSE.txt for details.

// EMG data collector with interval-based CSV logging
// Collects 3-second samples with 2-second heads up before each collection

#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <thread>

#include <myo/myo.hpp>

class IntervalDataCollector : public myo::DeviceListener {
public:
    enum class CycleState {
        HEADS_UP,      // 0-2s: Warning before collection
        COLLECTING     // 2-5s: Collecting data (3 seconds)
    };

    IntervalDataCollector(const std::string& csvFilename)
    : emgSamples()
    , csvFile(csvFilename)
    , sampleCount(0)
    , cycleStartTime(std::chrono::high_resolution_clock::now())
    , programStartTime(std::chrono::high_resolution_clock::now())
    , currentState(CycleState::HEADS_UP)
    , isClench(true)  // Start with clench
    {
        // Write CSV header
        if (csvFile.is_open()) {
            csvFile << "timestamp,sample_number,label,emg1,emg2,emg3,emg4,emg5,emg6,emg7,emg8\n";
            csvFile.flush();
        } else {
            throw std::runtime_error("Unable to open CSV file for writing");
        }
        
        // Print initial heads up
        std::cout << "\n=== HEADS UP: Get ready for " << (isClench ? "CLENCH" : "REST") << " in 2 seconds... ===" << std::endl;
    }
    
    ~IntervalDataCollector() {
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
        
        // Calculate time since cycle start
        auto currentTime = std::chrono::high_resolution_clock::now();
        double cycleElapsed = std::chrono::duration<double>(currentTime - cycleStartTime).count();
        double programElapsed = std::chrono::duration<double>(currentTime - programStartTime).count();
        
        // Update state based on cycle time
        CycleState newState = currentState;
        if (cycleElapsed < 2.0) {
            newState = CycleState::HEADS_UP;
        } else if (cycleElapsed < 5.0) {
            newState = CycleState::COLLECTING;
        } else {
            // Cycle complete, start new cycle
            cycleStartTime = currentTime;
            isClench = !isClench;  // Alternate label
            newState = CycleState::HEADS_UP;
            std::cout << "\n=== HEADS UP: Get ready for " << (isClench ? "CLENCH" : "REST") << " in 2 seconds... ===" << std::endl;
        }
        
        // Handle state transitions and notifications
        if (newState != currentState) {
            currentState = newState;
            
            if (currentState == CycleState::COLLECTING) {
                std::cout << "\n>>> COLLECTING " << (isClench ? "CLENCH" : "REST") << " data... <<<" << std::endl;
            }
        }
        
        // Only write to CSV during collection phase
        if (currentState == CycleState::COLLECTING && csvFile.is_open()) {
            csvFile << std::fixed << std::setprecision(6) << programElapsed << ","
                    << sampleCount << ","
                    << (isClench ? "clench" : "rest");
            for (int i = 0; i < 8; i++) {
                csvFile << "," << static_cast<int>(emg[i]);
            }
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
        std::cout << " [" << stateStr << " - " << (isClench ? "CLENCH" : "REST") << "]";
        
        // Print cycle time
        auto currentTime = std::chrono::high_resolution_clock::now();
        double cycleElapsed = std::chrono::duration<double>(currentTime - cycleStartTime).count();
        std::cout << " Cycle: " << std::fixed << std::setprecision(1) << cycleElapsed << "s";

        std::cout << std::flush;
    }

    uint64_t getSampleCount() const { return sampleCount; }

private:
    std::array<int8_t, 8> emgSamples;
    std::ofstream csvFile;
    uint64_t sampleCount;
    std::chrono::high_resolution_clock::time_point cycleStartTime;
    std::chrono::high_resolution_clock::time_point programStartTime;
    CycleState currentState;
    bool isClench;  // true for clench, false for rest
};

int main(int argc, char** argv)
{
    try {
        // Generate filename with timestamp
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::system_clock::to_time_t(now);
        std::stringstream filenameStream;
        filenameStream << "emg_data_interval_" << timestamp << ".csv";
        std::string filename = filenameStream.str();
        
        std::cout << "=====================================" << std::endl;
        std::cout << "EMG Interval Data Collector" << std::endl;
        std::cout << "CSV file: " << filename << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << "Cycle: 2s heads up -> 3s collection (alternates CLENCH/REST)" << std::endl;
        std::cout << "Alternates between CLENCH and REST labels" << std::endl;
        std::cout << "=====================================" << std::endl;
        
        myo::Hub hub("com.example.emg-interval-sample");

        std::cout << "Attempting to find a Myo..." << std::endl;

        myo::Myo* myo = hub.waitForMyo(10000);

        if (!myo) {
            throw std::runtime_error("Unable to find a Myo!");
        }

        std::cout << "Connected to a Myo armband!" << std::endl;

        // Enable EMG streaming at 200Hz
        myo->setStreamEmg(myo::Myo::streamEmgEnabled);

        // Create data collector with CSV file
        IntervalDataCollector collector(filename);

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

