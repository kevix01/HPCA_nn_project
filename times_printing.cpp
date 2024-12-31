//
// Created by kevin on 28/12/24.
//

#include <chrono> // For time-related functions and types
#include <iomanip> // For formatting output (e.g., std::setw)
#include <iostream> // For input/output operations
#include "times_printing.h" // Include the corresponding header file

// Function to print elapsed time in a human-readable format
void printElapsedTime(const std::chrono::duration<double>& elapsed_time) {
    // Convert the elapsed time into hours, minutes, seconds, milliseconds, and microseconds
    auto hours = std::chrono::duration_cast<std::chrono::hours>(elapsed_time).count(); // Extract hours
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(elapsed_time).count() % 60; // Extract minutes (mod 60 to get remaining minutes)
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed_time).count() % 60; // Extract seconds (mod 60 to get remaining seconds)
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() % 1000; // Extract milliseconds (mod 1000 to get remaining milliseconds)
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() % 1000; // Extract microseconds (mod 1000 to get remaining microseconds)

    // Print the elapsed time in a formatted table
    std::cout << "###############################\n"; // Print a separator line
    std::cout << std::setw(14) << "Hours:"        << std::setw(10) << hours        << "\n"; // Print hours
    std::cout << std::setw(14) << "Minutes:"      << std::setw(10) << minutes      << "\n"; // Print minutes
    std::cout << std::setw(14) << "Seconds:"      << std::setw(10) << seconds      << "\n"; // Print seconds
    std::cout << std::setw(14) << "Milliseconds:" << std::setw(10) << milliseconds << "\n"; // Print milliseconds
    std::cout << std::setw(14) << "Microseconds:" << std::setw(10) << microseconds << "\n"; // Print microseconds
    std::cout << "###############################\n"; // Print a separator line
}
