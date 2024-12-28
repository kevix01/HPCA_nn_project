//
// Created by kevin on 28/12/24.
//

#include <chrono>
#include <iomanip>
#include <iostream>
#include "times_printing.h"

void printElapsedTime(const std::chrono::duration<double>& elapsed_time) {
    auto hours = std::chrono::duration_cast<std::chrono::hours>(elapsed_time).count();
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(elapsed_time).count() % 60;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed_time).count() % 60;
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() % 1000;
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() % 1000;

    std::cout << "###############################\n";
    std::cout << std::setw(12) << "Hours:"        << std::setw(8) << hours        << "\n";
    std::cout << std::setw(12) << "Minutes:"      << std::setw(8) << minutes      << "\n";
    std::cout << std::setw(12) << "Seconds:"      << std::setw(8) << seconds      << "\n";
    std::cout << std::setw(12) << "Milliseconds:" << std::setw(8) << milliseconds << "\n";
    std::cout << std::setw(12) << "Microseconds:" << std::setw(8) << microseconds << "\n";
    std::cout << "###############################\n";
}
