import edgeiq


def main():
    obj_detect = edgeiq.ObjectDetection("alwaysai/mobilenet_ssd")
    if edgeiq.is_jetson():
        obj_detect.load(engine=edgeiq.Engine.DNN_CUDA)
    else:
        obj_detect.load(engine=edgeiq.Engine.DNN)

    print("Loaded model:\n{}\n".format(obj_detect.model_id))
    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Labels:\n{}\n".format(obj_detect.labels))

    fps = edgeiq.FPS()

    try:
        with edgeiq.FileVideoStream(
                "crosswalk.m4v", play_realtime=True) as video_stream, \
                        edgeiq.Streamer() as streamer:
            fps.start()

            # loop detection
            while True:
                try:
                    frame = video_stream.read()
                except edgeiq.NoMoreFrames:
                    # Restart video when it ends
                    video_stream.start()
                    continue
                results = obj_detect.detect_objects(frame, confidence_level=.5)

                frame = edgeiq.markup_image(
                        frame, results.predictions, colors=obj_detect.colors)

                # Generate text to display on streamer
                text = ["Model: {}".format(obj_detect.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration))
                text.append("Objects:")

                for prediction in results.predictions:
                    text.append("{}: {:2.2f}%".format(
                        prediction.label, prediction.confidence * 100))

                streamer.send_data(frame, text)

                fps.update()

                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
