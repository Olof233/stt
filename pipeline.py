import requests
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
import subprocess
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Chunk_timestamps:
    def __init__(self,
                 line = 0,
                 chunk=None,
                 start_time=None,
                 end_time=None):
        self.line = line
        self.chunk = chunk
        self.start_time = start_time
        self.end_time = end_time

    def getstamps(self):
        return {
                "end_time": self.end_time,
                "line": self.line,
                "text": self.chunk,
                "start_time": self.start_time
            }
    
def seconds_to_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


video = "D:\\desktop\\1\\work\\ai4c\\chunking\\test\\【克苏鲁神话】敦威治恐怖事件.mp4"

url = "http://127.0.0.1:9977/api"
# 请求参数  file:音视频文件，language：语言代码，model：模型，response_format:text|json|srt
# 返回 code==0 成功，其他失败，msg==成功为ok，其他失败原因，data=识别后返回文字
files = {"file": open(video, "rb")}
data={"language":"zh","model":"tiny","response_format":"json","word_timestamps":"true"}
logger.info("Ectracting document")
response = requests.request("POST", url, timeout=600, data=data,files=files)

chunks = []
for i in response.json()["data"]:
        chunks.append({'text':i["text"].strip("。") + '。'})

document = ""
for i in chunks:
    document += i["text"]

logger.info("Segmenting document")
p = pipeline(
    task=Tasks.document_segmentation,
    model='modelscope/Seqmodel', model_revision='master')
logger.info("Segmentation done")

result = p(document)

chunks = [chunk.strip("\t") for chunk in result[OutputKeys.TEXT].split("\n") if len(chunk.strip("\t")) > 0] # type: ignore

chunk_timestamps = []
line = 0
with tqdm(total=len(chunks), desc="Matching timestamps: ", unit="chunk") as pbar:
    for chunk in chunks:
        newchunk = Chunk_timestamps(chunk=chunk, line=line)
        found1 = False
        found2 = False
        for entry in response.json()["data"]:
            if chunk.split("。")[0] in entry["text"]:
                for word in entry["words"]:
                    if chunk.split("。")[0][0] in word["word"]:
                        newchunk.start_time = seconds_to_srt_time(word["start"])
                        found1 = True
            if chunk.split("。")[-2] in entry["text"]:
                for word in entry["words"]:
                    if chunk.split("。")[-2][-1] in word["word"]:
                        newchunk.end_time = seconds_to_srt_time(word["end"])
                        found2 = True
            if found1 and found2:
                found1 = False
                found1 = False
                break
        if not found1 and not found2:
            chunk_timestamps.append({
                "end_time": None,
                "text": chunk,
                "start_time": None
            })
        chunk_timestamps.append(newchunk)
        line += 1
        pbar.update(1)

cnt = 0
with tqdm(total=len(chunk_timestamps), desc="Processing videos: ", unit="chunk") as pbar:
    for chunk_timestamp in chunk_timestamps:
        info = chunk_timestamp.getstamps()
        start_time = info["start_time"]
        end_time = info["end_time"]
        output = "test/test" + str(cnt) + ".mp4"
        ffmpeg_command1 = "D:/desktop/1/work/ai4c/stt/ffmpeg -y -i " +  video
        ffmpeg_command2= " -ss " + start_time
        ffmpeg_command3 = " -to " + end_time + " -c copy"
        ffmpeg_command4 = " -vcodec libx264 -acodec libmp3lame " + output
        ffmpeg_command = ffmpeg_command1 + ffmpeg_command2 + ffmpeg_command3 + ffmpeg_command4
        p = subprocess.Popen(ffmpeg_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        cnt += 1
        pbar.update(1)