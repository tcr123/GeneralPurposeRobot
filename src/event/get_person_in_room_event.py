import asyncio

from command.face_detection_command import FaceDetectionCommand
from command.face_recognition_command import FaceRecognitionCommand
from command.record_video_command import RecordVideoCommand
from event.event_class import EventClass

class GetPersonInRoomEvent(EventClass):
    def __init__(self):
        super().__init__()
        self.add_commands()

    def add_commands(self) -> None:
        self.add_cmd(FaceDetectionCommand(), "concurrent")
        self.add_cmd(FaceRecognitionCommand(), "concurrent")
        self.add_cmd(RecordVideoCommand(), "concurrent")
