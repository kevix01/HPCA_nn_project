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
    std::cout << std::setw(14) << "Hours:"        << std::setw(10) << hours        << "\n";
    std::cout << std::setw(14) << "Minutes:"      << std::setw(10) << minutes      << "\n";
    std::cout << std::setw(14) << "Seconds:"      << std::setw(10) << seconds      << "\n";
    std::cout << std::setw(14) << "Milliseconds:" << std::setw(10) << milliseconds << "\n";
    std::cout << std::setw(14) << "Microseconds:" << std::setw(10) << microseconds << "\n";
    std::cout << "###############################\n";
}
