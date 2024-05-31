import os
import shutil


def transferAudio(sourceVideo, targetVideo):

    tempAudioFileName = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:

        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        os.system(
            'ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName)
        )

    print(targetVideo)

    targetNoAudio = (
        os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    )
    os.rename(targetVideo, targetNoAudio)
    # combine audio file and new video file
    os.system(
        'ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(
            targetNoAudio, tempAudioFileName, targetVideo
        )
    )

    if (
        os.path.getsize(targetVideo) == 0
    ):  # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system(
            'ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(
                sourceVideo, tempAudioFileName
            )
        )
        os.system(
            'ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(
                targetNoAudio, tempAudioFileName, targetVideo
            )
        )
        if (
            os.path.getsize(targetVideo) == 0
        ):  # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print(
                "Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead."
            )

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")
