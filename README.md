# Gait Pattern Recognition with Arduino Nano 33 BLE Sense's IMU

This repository contains the implementation and resources for **Gait Pattern Recognition** using the **Arduino Nano 33 BLE Sense's IMU**. The project involves capturing human gait patterns using the IMU sensor embedded in the Arduino Nano 33 BLE Sense, applying machine learning algorithms, and recognizing different gait patterns.

## Table of Contents

- [Project Structure](#project-structure)
- [Directory Description](#directory-description)
- [Authors](#authors)

---

## Project Structure

The project is structured as follows:

| Directory                  | Description                                           |
|----------------------------|-------------------------------------------------------|
| [Assembly](./Assembly)      | Hardware assembly for the Arduino and IMU setup.      |
| [Code](./Code)              | Scripts for data collection and ML model training.    |
| [GaitPattern](./GaitPattern)| Gait datasets (raw and preprocessed).                 |
| [Manual](./Manual)          | User manual for setup and operation.                  |
| [MLlib](./MLlib)            | Machine learning libraries and tools.                 |
| [Nano33BLESense](./Nano33BLESense) | Arduino Nano 33 BLE Sense documentation.     |
| [Poster](./Poster)          | Research poster for presentations.                    |
| [Presentations](./Presentations) | Project presentations (slides, PDFs, etc.).     |
| [ProjectManagement](./ProjectManagement) | Project timelines, task lists, etc.   |
| [Report](./report)          | Final project report.                                 |
| [.gitignore](./.gitignore)  | Files and folders to ignore in version control.       |
| [README.md](./README.md)    | This file, the project overview.                      |
| [author](./author)          | Author details and contribution records.              |

---

## Directory Description

### Assembly

[Assembly](./Assembly) contains details regarding the hardware assembly of the Arduino Nano 33 BLE Sense, along with the steps to configure the IMU sensor for data collection.

### Code

[Code](./Code) includes all the scripts and programs used for data collection, preprocessing, and machine learning model training. These scripts allow the Arduino to collect IMU data and perform inference on gait patterns.

### GaitPattern

[GaitPattern](./GaitPattern) holds the datasets generated from the IMU sensors, including preprocessed and raw gait data. These datasets are used to train the ML model.

### Manual

[Manual](./Manual) contains the user manual for setting up the hardware and running the software for gait pattern recognition.

### MLlib

[MLlib](./MLlib) is the library of machine learning algorithms used in this project. It contains various Python scripts and tools required for data analysis, feature extraction, and model building.

### Nano33BLESense

[Nano33BLESense](./Nano33BLESense) provides details on the Arduino Nano 33 BLE Sense board, including sensor specifications, setup guides, and firmware configuration.

### Poster

[Poster](./Poster) contains the research poster used for presenting this project at various conferences or project demonstrations.

### Presentations

[Presentations](./Presentations) includes PowerPoint or PDF presentations related to the project’s research and findings.

### ProjectManagement

[ProjectManagement](./ProjectManagement) holds the project management documents such as timelines, task assignments, and status reports.

### Report

[Report](./report) contains the final project report, detailing the methodology, data analysis, results, and conclusions drawn from the gait pattern recognition experiments.

---

## Authors

- Abishek
- Fedor
- Bruna

---

Feel free to explore each section, and use the `Code` directory to run the project or contribute improvements!
