import gc 
import os
import argparse
import sys
import uvicorn
import torch
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# 将当前目录添加到 sys.path 以确保 indextts 模块可以被正确导入
sys.path.append(os.getcwd())

from indextts.infer_v2 import IndexTTS2

# 参数解析 (尽可能与 webui.py 保持一致)
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="checkpoints", help="模型目录路径")
parser.add_argument("--port", type=int, default=8080, help="服务器端口")
parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
parser.add_argument("--fp16", action="store_true", help="使用 FP16 浮点数")
parser.add_argument("--deepspeed", action="store_true", help="使用 DeepSpeed 加速")
parser.add_argument("--cuda_kernel", action="store_true", help="使用 CUDA 内核加速")
parser.add_argument("--output_dir", type=str, default="outputs", help="生成音频的输出目录")
cmd_args = parser.parse_args()

# 初始化 FastAPI 应用
app = FastAPI(title="IndexTTS2 API 服务")

# 配置 CORS (跨域资源共享)
# 允许所有来源访问，以便本地 HTML 文件可以顺利调用 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 TTS 模型
print(f"正在从目录 {cmd_args.model_dir} 加载 IndexTTS2 模型...")
tts = IndexTTS2(
    model_dir=cmd_args.model_dir,
    cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
    use_fp16=cmd_args.fp16,
    use_deepspeed=cmd_args.deepspeed,
    use_cuda_kernel=cmd_args.cuda_kernel,
)
print("模型加载成功。")

# 确保输出目录存在
os.makedirs(cmd_args.output_dir, exist_ok=True)
# 确保上传目录存在 (如有需要)
UPLOAD_DIR = "uploaded_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 数据模型定义
class SynthesizeRequest(BaseModel):
    text: str
    audio_path: str  # 说话人/情绪的参考音频路径
    emo_vector: Optional[List[float]] = None

# --- API 接口定义 ---

@app.get("/")
def read_root():
    return {"message": "IndexTTS2 API 服务器正在运行"}

@app.get("/v1/check/audio")
def check_audio(file_name: str = Query(..., description="要检查的音频文件的完整路径或文件名")):
    """
    检查服务器上是否存在指定的音频文件。
    HTML 客户端会发送 'file_name'，这可能是一个绝对路径，也可能是相对路径。
    """
    # 安全检查：根据需要可以防止目录遍历，但在本地工具中我们信任此路径
    # 如果是绝对路径，直接检查
    # index-tts 通常会在其运行目录或特定文件夹中查找相关文件
    
    exists = os.path.exists(file_name)
    return {"exists": exists}

@app.post("/v1/upload_audio")
async def upload_audio(
    audio: UploadFile = File(...),
    full_path: str = Form(...)
):
    """
    上传音频文件到指定路径。
    HTML 客户端会发送 'full_path'，指明文件应该被保存在哪里。
    """
    try:
        # 如果目录不存在，则创建目录
        directory = os.path.dirname(full_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # 确定保存位置
        save_path = full_path
        
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
            
        return {"status": "success", "path": save_path}
    except Exception as e:
        print(f"上传错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v2/synthesize")
def synthesize(req: SynthesizeRequest):
    try:
        if not os.path.exists(req.audio_path):
             raise HTTPException(status_code=400, detail=f"未找到参考音频文件: {req.audio_path}")

        # 生成唯一的输出文件名
        timestamp = int(os.times().elapsed * 1000)
        out_filename = f"gen_{timestamp}.wav"
        output_path = os.path.join(cmd_args.output_dir, out_filename)

        print(f"正在合成: 文本='{req.text}'")

        # --- 优化点 1: 使用 torch.no_grad() 强制关闭梯度 ---
        with torch.no_grad():
            tts.infer(
                spk_audio_prompt=req.audio_path,
                text=req.text,
                output_path=output_path,
                emo_audio_prompt=req.audio_path, 
                emo_vector=req.emo_vector,
                emo_alpha=1.0,
                use_emo_text=False,
                use_random=False,
            )

        if not os.path.exists(output_path):
             raise HTTPException(status_code=500, detail="合成失败: 输出文件未创建。")

        # --- 优化点 2: 显式清理垃圾和显存缓存 ---
        gc.collect() 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return FileResponse(output_path, media_type="audio/wav", filename=out_filename)

    except Exception as e:
        print(f"合成错误: {e}")
        # 即使出错也尝试清理
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))
        # 调用 IndexTTS2 推理
        # 注意: infer_v2.py 中的 'infer' 方法签名:
        # infer(self, spk_audio_prompt, text, output_path, emo_audio_prompt=None, emo_alpha=1.0, emo_vector=None, ...)
        
        # 我们使用传入的 audio_path 同时作为说话人提示 (speaker prompt) 和情绪提示 (emotion prompt)
        # 除非在更复杂的场景下需要区分
        
        tts.infer(
            spk_audio_prompt=req.audio_path,
            text=req.text,
            output_path=output_path,
            emo_audio_prompt=req.audio_path, # 同时也用作情绪参考
            emo_vector=req.emo_vector,
            # 默认参数
            emo_alpha=1.0,
            use_emo_text=False,
            use_random=False,
        )

        if not os.path.exists(output_path):
             raise HTTPException(status_code=500, detail="合成失败: 输出文件未创建。")

        return FileResponse(output_path, media_type="audio/wav", filename=out_filename)

    except Exception as e:
        print(f"合成错误: {e}")
        # traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=cmd_args.host, port=cmd_args.port)
