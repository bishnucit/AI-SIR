import asyncio
import os
import re
import subprocess
import tempfile
import requests
import edge_tts

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

OLLAMA_MODEL = "gemma4:latest"

VOICE_A = "en-US-GuyNeural"
VOICE_B = "en-US-JennyNeural"

TEMP_DIR = tempfile.mkdtemp(prefix="podcast_")

FFMPEG_PATH = r"REPLACE THIS WITH PATH TO YOUR FFMPEG - you can find it by using 'where FFMPEG' in cmd prompt"


# --------------------------------------------------
# READ MARKDOWN
# --------------------------------------------------

def read_markdown(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# --------------------------------------------------
# GENERATE SCRIPT
# --------------------------------------------------

def generate_script(markdown):
    system_instruction = (
        "You are generating a conversational podcast script between two hosts: Alice and Bob.\n"
        "Strict formatting rules:\n"
        "- Format every single dialogue line starting with 'Alice: ' or 'Bob: '\n"
        "- Do not write any markdown code fences, headers, introductions, or sign-offs."
    )
    
    try:
        # Standard Chat Endpoint with expanded context window capacity
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"Generate a long podcast based on this text:\n\n{markdown}"}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_ctx": 16384  
                }
            },
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        if content and content.strip():
            return content
    except Exception as e:
        print(f"Chat endpoint failed ({e}). Attempting raw generation fallback...")

    try:
        # Emergency Fallback Endpoint (if api/chat handles formatting poorly)
        prompt = f"{system_instruction}\n\nInput Document:\n{markdown}\n\nGenerate Script:"
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_ctx": 16384}
            },
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"Fallback generation failed: {e}")
        return ""


# --------------------------------------------------
# PARSE DIALOG
# --------------------------------------------------

def parse_dialog(script):
    dialog = []
    
    if not script or not str(script).strip():
        return dialog

    clean_script = str(script).replace("**", "").replace("*", "")
    clean_script = re.sub(r"```[a-zA-Z]*", "", clean_script)
    
    for line in clean_script.splitlines():
        line = line.strip()
        if not line:
            continue
            
        # FIXED: Changed re.ignorecase to re.IGNORECASE
        m = re.match(r"^[\s*\[]*?(alice|bob|speaker\s*a|speaker\s*b|host|guest|a|b)[\s*\]]*?\s*[:\-]\s*(.+)", line, re.IGNORECASE)
        if m:
            tag = m.group(1).lower()
            text = m.group(2).strip()
            speaker = "Alice" if tag in ["alice", "speaker a", "host", "a"] else "Bob"
            dialog.append((speaker, text))

    # Alternate line assignment safety net
    if not dialog:
        print("Standard parsing pattern matched zero lines. Assigning turns alternatingly...")
        toggle = True
        for line in clean_script.splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "title:", "podcast:", "---")):
                continue
            speaker = "Alice" if toggle else "Bob"
            dialog.append((speaker, line))
            toggle = not toggle

    print(f"Successfully compiled {len(dialog)} conversation turns.")
    return dialog


# --------------------------------------------------
# GENERATE AUDIO
# --------------------------------------------------

async def generate_audio(dialog):
    files = []

    for i, (speaker, text) in enumerate(dialog):
        try:
            voice = VOICE_A if speaker == "Alice" else VOICE_B
            filename = os.path.join(TEMP_DIR, f"{i:04d}.mp3")

            communicate = edge_tts.Communicate(text=text, voice=voice)
            await communicate.save(filename)

            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                files.append(filename)

        except Exception as e:
            print(f"Skipping line {i} due to TTS error: {e}")
            pass

    return files


# --------------------------------------------------
# MERGE AUDIO
# --------------------------------------------------

def merge_audio(files, output):
    ffmpeg = FFMPEG_PATH

    if not os.path.exists(ffmpeg):
        raise RuntimeError(f"FFmpeg not found: {ffmpeg}")

    if not files:
        raise RuntimeError("No audio files generated")

    concat_file = os.path.join(TEMP_DIR, "concat.txt")

    with open(concat_file, "w", encoding="utf-8") as f:
        for file in files:
            if os.path.exists(file):
                path = os.path.abspath(file).replace("\\", "/")
                f.write(f"file '{path}'\n")

    cmd = [
        ffmpeg,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-acodec", "libmp3lame",
        "-ar", "44100",
        "-b:a", "192k",
        output,
    ]

    subprocess.run(cmd, check=True)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

async def main():
    md_file = "results/test1.md"
    output = "results/podcast.mp3"

    os.makedirs(os.path.dirname(output), exist_ok=True)

    print(f"Reading source file: {md_file}...")
    markdown = read_markdown(md_file)

    print("Generating podcast script from Ollama (this might take a minute)...")
    script = generate_script(markdown)

    print("Processing script dialogue layout...")
    dialog = parse_dialog(script)

    if not dialog:
        print("\n--- DEBUG: RAW LLM OUTPUT RECEIVED ---")
        print(repr(script))
        print("---------------------------------------\n")
        raise RuntimeError("The local LLM returned an empty response or processing failed.")

    print("Converting text lines to high-fidelity audio streams via edge-tts...")
    files = await generate_audio(dialog)

    if not files:
        raise RuntimeError("No audio files were generated successfully")

    print("Stitching individual dialogue segments together via FFmpeg...")
    merge_audio(files, output)

    print("DONE! File saved at:", output)


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()

    asyncio.run(main())
