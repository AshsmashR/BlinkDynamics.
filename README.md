

# 👁️Blink Detection and Drowsiness Monitoring👁️
<p align="left">
  <img src="https://img.shields.io/badge/Python-3.8%2B-black?style=flat&logo=python">
  <img src="https://img.shields.io/badge/OpenCV-Vision-black?style=flat&logo=opencv">
  <img src="https://img.shields.io/badge/NumPy-Scientific-black?style=flat&logo=numpy">
  <img src="https://img.shields.io/badge/Status-Research-black?style=flat">
  <img src="https://img.shields.io/badge/Method-EAR%20%7C%20PERCLOS-black?style=flat">
</p>

A lightweight, research-oriented blink detection pipeline for experimental drowsiness analysis using eye dynamics, blink rate, blink duration, and PERCLOS.

## Overview

This repository implements a vision-based blink monitoring system that tracks eye-state changes over time to estimate fatigue. It computes blink frequency, average blink duration, and PERCLOS, and summarizes these metrics at fixed checkpoints to provide interpretable alertness states.

Example output:

```
Checkpoint 85s:
Blink Rate: 14.67/min | PERCLOS: 12.17% | Avg Duration: 224ms → Alert / Questionable

Checkpoint 115s:
Blink Rate: 18.40/min | PERCLOS: 11.93% | Avg Duration: 206ms → Drowsy / Questionable
```

## Limitations

The current approach relies on heuristic signals (dark ratio and eye openness), which can be unreliable for individuals with smaller eyes, under-eye shadows, or darker periocular regions. Performance is also sensitive to lighting, pose, and natural variation in blinking behavior.

## Purpose

This repository serves as an interpretable baseline for blink-based fatigue analysis and as a foundation for building more robust, multimodal drowsiness detection systems.

---



