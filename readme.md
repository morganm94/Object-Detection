# Crosswalk Object Detection
This app performs object detection on a video of a crosswalk in simulated real-time. The app will automatically perform inference on the CUDA GPU on Jetson devices.

## Setup
This app requires an alwaysAI account. Head to the [Sign up page](https://www.alwaysai.co/dashboard) if you don't have an account yet. Follow the instructions to install the alwaysAI tools on your development machine.

Next, create an empty project to be used with this app. When you clone this repo, you can run `aai app configure` within the repo directory and your new project will appear in the list.

## Usage
Once the alwaysAI tools are installed on your development machine (or edge device if developing directly on it) you can run the following CLI commands:

To set up the target device and install path

```
aai app configure
```

To build and deploy the docker image of the app to the target device

```
aai app install
```

To start the app with a video stream

```
aai app start
```

