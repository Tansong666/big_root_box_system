from PyQt5.QtCore import QThread, pyqtSignal,QThreadPool
import wave
from piper import PiperVoice, SynthesisConfig
# 1. 导入依赖（新增播放所需库）
import pygame
import serial
import time


class TextToSpeech(QThread):
    def __init__(self, model_path = "E:\\big_root_system\\models\\tts\\chinese\\zh_CN-huayan-medium.onnx", syn_config = None):
        super().__init__()
        self.model_path = model_path
        self.syn_config = SynthesisConfig(
            volume=1,  # half as loud
            length_scale=2,  # twice as slow  0.8
            noise_scale=0.767,  # more audio variation
            noise_w_scale=0.8,  # more speaking variation
            normalize_audio=False,  # use raw audio from voice
        )
        self.voice = PiperVoice.load(self.model_path)

        # 2. 新增：初始化音频播放器（适配Piper合成的音频格式）
    def init_pygame_player(self, sample_rate: int):
        """初始化pygame音频播放器，参数匹配Piper合成的音频格式"""
        # Piper固定输出：16位单声道PCM音频，因此size设为-16（负号表示有符号整数）
        pygame.mixer.init(
            frequency=sample_rate,  # 采样率，从Piper模型配置中获取
            size=-16,
            channels=1,
            buffer=2048  # 播放缓冲区（越小延迟越低，2048~4096为推荐值）
        )
        # 等待播放器初始化完成
        while not pygame.mixer.get_init():
            pass


    # 3. 新增：逐块播放合成的音频
    def play_audio_chunks(self, audio_chunks):
        """播放voice.synthesize()生成的音频块迭代器"""
        first_chunk = True  # 标记是否为第一个音频块（用于初始化播放器）

        for chunk in audio_chunks:
            # 第一个音频块：获取采样率并初始化播放器
            if first_chunk:
                self.init_pygame_player(sample_rate=chunk.sample_rate)
                first_chunk = False

            # 播放当前音频块（使用16位PCM字节数据，Piper已封装在audio_int16_bytes属性中）
            sound = pygame.mixer.Sound(buffer=chunk.audio_int16_bytes)
            sound.play()

            # 等待当前块播放完成，避免多块重叠
            while pygame.mixer.get_busy():
                pygame.time.Clock().tick(10)  # 降低CPU占用

        # 播放完成后清理播放器资源
        pygame.mixer.quit()
    
    def syn_and_play(self, text, syn_config=None):
        self.text = text
        audio_chunks = self.voice.synthesize(self.text, syn_config)  # 生成音频块迭代器
        self.play_audio_chunks(audio_chunks)

    def open_serial(self):
        self.ser = serial.Serial("com5", 115200)  # "/dev/ttyUSB0"
        if self.ser.isOpen():
            print("Speech Serial Opened! Baudrate=115200")
        else:
            print("Speech Serial Open Failed!")

    def void_write(self, void_data=0x63):
        hex_string = int(void_data)
        cmd = [0xAA, 0x55, 0xFF, hex_string,0xFB]
        self.ser.write(cmd)  # 发送后自动播报
        time.sleep(0.005)
        self.ser.flushInput()

    def speech_read(self):
        count = self.ser.inWaiting()
        if count:
            speech_data = self.ser.read(count)
            hex_data = speech_data.hex()
            print(f"Read hex data: {hex_data}")
            if hex_data.startswith('aa55'):
                # byte1 = hex_data[4:6]  # 提取 '00'
                byte2 = hex_data[6:8]  # 提取 '00'
                if byte2 == '01':
                        self.ser.flushInput()
                        time.sleep(0.005)
                        self.syn_and_play("好的，已停止",self.syn_config)
                elif byte2 == '02':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("好的，已停车")
                elif byte2 == '03':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("好的，正在前进")
                elif byte2 == '04':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("好的，正在后退")
                elif byte2 == '05':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("好的，正在加速")
                elif byte2 == '06':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("好的，正在减速")
                elif byte2 == '07':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("好的，正在后退三秒")
                elif byte2 == '08':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("好的，已关灯")
                elif byte2 == '09':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("好的，已亮一三灯")
                elif byte2 == '0a':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("好的，已亮二四灯")
                # 变量
                elif byte2 == '0b':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("好的，已息屏")
                elif byte2 == '0c':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("好的，已亮屏")
                elif byte2 == '0d':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("当前电量百分之多少")
                    # with wave.open("chinese.wav", "wb") as wav_file:
                    #     self.voice.synthesize_wav("当前电量百分之多少", wav_file)
                elif byte2 == '0e':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    if self.ser.isOpen():
                        self.syn_and_play("连接成功")
                    else:
                        self.syn_and_play("连接失败")
                elif byte2 == '0f':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("当前保存路径是什么")
                elif byte2 == '10':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("当前磁盘空间还剩百分之多少")
                elif byte2 == '11':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("当前二维码位置为多少")
                elif byte2 == '12':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("好的，已加载")
                elif byte2 == '13':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("好的，开始采集")
                elif byte2 == '14':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("好的，开始采集和处理")
                elif byte2 == '15':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("目前采集了多少张图像")
                elif byte2 == '16':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("当前根系的长度为多少")
                elif byte2 == '17':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("对照组根系表型")
                elif byte2 == '18':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("干旱组根系表型")
                elif byte2 == '19':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("好的，正在上传根系数据")
                elif byte2 == '20':
                    self.ser.flushInput()
                    time.sleep(0.005)
                    self.syn_and_play("好的，开始分析")

    def run(self):
        self.open_serial()
        self.void_write(0x63)
        time.sleep(0.005)
        while True:
            self.speech_read()
            
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    # app = QApplication([])
    thread =TextToSpeech()
    thread.start()
    print("语音线程已启动。输入 'exit' 退出程序。")
    while True:
        cmd = input().strip().lower()
        if cmd == 'exit':
            thread.stop()
            break
     
    sys.exit(app.exec_())



    
