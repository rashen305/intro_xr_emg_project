// Copyright (C) 2013-2014 Thalmic Labs Inc.
// Distributed under the Myo SDK license agreement. See LICENSE.txt for details.

// EMG data collector with CSV logging at 200Hz

#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>

#include <myo/myo.hpp>

class DataCollector : public myo::DeviceListener {
public:
    DataCollector(const std::string& csvFilename)
    : emgSamples()
    , csvFile(csvFilename)
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
        
        sampleCount++;
    }

    // Print the current EMG values (like the original script)
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

        std::cout << std::flush;
    }

    uint64_t getSampleCount() const { return sampleCount; }

private:
    std::array<int8_t, 8> emgSamples;
    std::ofstream csvFile;
    uint64_t sampleCount;
    std::chrono::high_resolution_clock::time_point startTime;
};

int main(int argc, char** argv)
{
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
        std::cout << "=====================================" << std::endl;
        
        myo::Hub hub("com.example.emg-data-sample");

        std::cout << "Attempting to find a Myo..." << std::endl;

        myo::Myo* myo = hub.waitForMyo(10000);

        if (!myo) {
            throw std::runtime_error("Unable to find a Myo!");
        }

        std::cout << "Connected to a Myo armband!" << std::endl;

        // Enable EMG streaming at 200Hz
        myo->setStreamEmg(myo::Myo::streamEmgEnabled);

        // Create data collector with CSV file
        DataCollector collector(filename);

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