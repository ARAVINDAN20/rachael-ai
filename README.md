
# Rachel HR Interview Bot
<img src="https://github.com/user-attachments/assets/ec395b1f-6acd-4923-8e4c-d67a4b48271d" alt="Rachel HR Interview Bot" width="300" height="360">

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [Installation](#installation)
   - [Installation via Python pip](#installation-via-python-pip-)
   - [Quick Start with Docker](#quick-start-with-docker-)
   - [Other Installation Methods](#other-installation-methods)
   - [Troubleshooting](#troubleshooting)
   - [Keeping Your Docker Installation Up-to-Date](#keeping-your-docker-installation-up-to-date)
5. [Usage](#usage)
6. [Technical Details](#technical-details)
7. [Contributing](#contributing)
8. [License](#license)
9. [Acknowledgments](#acknowledgments)

## 1. Introduction

Rachel HR Interview Bot is a cutting-edge, AI-powered interview preparation assistant designed to revolutionize the HR interview process for students, freshers, and experienced professionals. Developed by Aravindan M (URK22AI1026) from the B.Tech Artificial Intelligence and Data Science program at Karunya Institute of Technology and Sciences, Rachel employs advanced natural language processing (NLP) techniques to provide a personalized and comprehensive interview experience.

NVIDIA's advanced AI technologies and OpenWebUI for a comprehensive interview preparation experience. Developed for the NVIDIA and LlamaIndex Developer Contest, this project combines state-of-the-art language models with robust security features and efficient processing capabilities

This README file provides an in-depth overview of the project, its features, technical details, and instructions for setup and usage.
## YouTube Demo Video
[Rachel: AI-Powered HR Interview Bot | NVIDIA & LlamaIndex Developer Contest Submission](https://youtu.be/s2ics4GZd-Q?si=nuwlVpq7QLkHbGPr) 
## 2.🌟 Key Features
Rachel HR Interview Bot offers a wide range of features designed to enhance the interview preparation process:

- **Advanced AI-Powered Interviews**
  - Utilizing NVIDIA's Nemotron-70B model for human-like interactions
  - Real-time response generation with TensorRT-LLM optimization
  - Context-aware question generation based on resume analysis

- **Comprehensive Security**
  - NeMo Guardrails integration for content safety
  - Professional tone maintenance
  - Ethical AI guidelines compliance

- **Smart Document Processing**
  - Resume analysis 
  - Technical skill extraction
  - Domain-specific question generation

- **Performance Optimization**
  - GPU acceleration with NVIDIA TensorRT-LLM
  - Efficient batch processing
  - Low-latency response generation

- **Interactive User Interface**
  - OpenWebUI integration for seamless experience
  - Real-time feedback system
  - Progress tracking
  - Interview simulation environment

## 🔧 Technical Architecture

```
+------------------------+     +----------------------+     +-------------------------+
|     User Interface     |     |    Core Engine      |     |    NVIDIA Services     |
|     (OpenWebUI)        | <-> |   (Python/Flask)    | <-> |  (NIM/TensorRT-LLM)   |
+------------------------+     +----------------------+     +-------------------------+
          ↑                            ↑                             ↑
          |                            |                             |
          v                            v                             v
+------------------------+     +----------------------+     +-------------------------+
|   Document Processor   |     |  Security Layer     |     |    Knowledge Base      |
| (NeMo Curator/Reader) |     | (NeMo Guardrails)   |     |   (LlamaIndex/NeMo)   |
+------------------------+     +----------------------+     +-------------------------+
```
![NVIDIA-LLAMAINDEX-CONTEST-ARCHITECTURE - Page 1](https://github.com/user-attachments/assets/f8c14d85-b0ce-44fa-9109-2ec54258f2cc)

## 📋 Requirements

- Python 3.11+
- NVIDIA GPU with CUDA support
- Docker (optional)
- 16GB+ RAM recommended
- 50GB+ disk space

## 🚀 Installation

### Method 1: Direct Installation

```bash
# Clone the repository
git clone https://github.com/ARAVINDAN20/rachael-ai.git
cd rachael-ai

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install nemo-guardrails tensorrt-llm nvidia-nim gradio spacy pytextrank scikit-learn openai
python -m spacy download en_core_web_sm

# Install NVIDIA components
pip install nvidia-pytriton
pip install nvidia-tensorrt
pip install nemo-toolkit[all]
pip install llama-index

#TO RUN THIS DEMO RACHEL IN GRADIO:
python Rachel.py
#THE MAIN IS OpenWebUI Follow the steps below 
```

### Method 2: Docker Installation

```bash
# Pull and run with GPU support
docker run -d --gpus all \
  -p 3000:8080 \
  -v rachel-data:/app/data \
  -e NVIDIA_API_KEY=your_key \
  --name rachel-hr-bot \
  ghcr.io/yourusername/rachel-hr-bot:latest
```

## 🔨 Core Components Integration

### 1. NVIDIA NeMo Guardrails Setup

```python
from nemo_guardrails import RailsConfig, Guard

class InterviewGuard:
    def __init__(self):
        self.config = RailsConfig.from_content("""
            define rail interview_safety:
                - maintain professional tone
                - ensure technical accuracy
                - avoid discriminatory content
                - provide constructive feedback
        """)
        self.guard = Guard(self.config)
```

### 2. NVIDIA NIM Integration

```python
from nvidia import nim

class ResponseGenerator:
    def __init__(self, api_key):
        self.client = nim.Client(
            base_url="https://api.nvidia.com/v1/nim",
            api_key=api_key
        )

    async def generate_response(self, prompt):
        return await self.client.generate(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            prompt=prompt,
            max_tokens=1024
        )
```

### 3. TensorRT-LLM Optimization

```python
import tensorrt_llm as trt_llm

class InferenceOptimizer:
    def __init__(self):
        self.engine = trt_llm.Engine(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            precision="fp16",
            max_batch_size=8
        )

    def optimize_inference(self, input_text):
        return self.engine.generate(
            input_text,
            max_length=512,
            temperature=0.7
        )
```

### 4. OpenWebUI Integration

```python
from fastapi import FastAPI
from openwebui import WebUI

app = FastAPI()
webui = WebUI(title="Rachel HR Bot")

@app.get("/")
async def root():
    return webui.render(
        template="interview",
        context={
            "mode": "interview",
            "features": ["resume_upload", "real_time_feedback"]
        }
    )
```

## 📝 Usage Guide

1. **Start the Application**
   [Open WebUI Documentation](https://docs.openwebui.com/troubleshooting/) run this Open WebUI in your system 
import this [rachel_json_file](https://drive.google.com/file/d/1tJqQUfS0smNcXtZxw1BG4dO09uBbHKUA/view?usp=drive_link)  (or)
   ```bash
   #to run this Demo project in Gradio and connect your API (https://build.nvidia.com/nvidia/llama-3_1-nemotron-70b-instruct)
   python Rachel.py
   ```

2. **Access the Interface**
   - Open your browser and navigate to `http://localhost:3000`
   - Log in with your credentials

3. **Upload Resume**
   - Click "Upload Resume" button
   - Select your PDF resume file
   - Wait for analysis completion

4. **Start Interview**
   - Choose your target role
   - Select technical domains
   - Begin interview session

5. **During Interview**
   - Answer questions naturally
   - Receive real-time feedback
   - Use hints when needed
   - Track your progress


1. **Resume Analysis**: Utilizes PDF extraction and NLP techniques to analyze resumes and identify domains of specialization.
2. **Personalized Question Generation**: Creates tailored technical HR interview questions based on the candidate's background, projects, and chosen job role.
3. **Interactive Chat Interface**: Provides a user-friendly for a seamless interview simulation experience.
4. **Answer Evaluation**: Employs advanced algorithms to assess user responses and provide ratings on a scale of 0-10.
5. **Constructive Feedback**: Offers detailed feedback on each answer, highlighting strengths and areas for improvement.
6. **Expected Answer Generation**: Provides model answers to help users understand ideal responses to interview questions.
7. **GPU Acceleration**: Utilizes CUDA(NVIDIA NIM Integration AIP) for faster processing and improved performance.
8. **Customizable Job Roles**: Supports a wide range of job roles across various engineering and scientific disciplines.
9. **Job Description Integration**: Incorporates specific job descriptions to generate highly relevant interview questions.
10. **Chat History Backup**: Allows users to save and review their interview sessions for future reference.

## 3. Technology Stack

Rachel HR Interview Bot leverages a powerful combination of cutting-edge technologies:

### NVIDIA AI Technologies
- **NVIDIA NeMo Guardrails**: For content safety and response filtering
- **TensorRT-LLM**: Hardware-accelerated inference optimization
- **NVIDIA NIM (NVIDIA Inference Microservices)**: Scalable API services
- **NVIDIA LLama 3.1 Nemotron 70B**: Core language model for question generation and evaluation

### Additional Technologies
- **SpaCy & PyTextRank**: Natural Language Processing
- **scikit-learn**: Machine Learning utilities
- **OpenAI API**: Integration with NVIDIA's API endpoint
- **PyPDF2**: PDF processing for resume analysis



## 4. Installation

### Installation via Python pip 🐍

Open WebUI can be installed using pip, the Python package installer. Before proceeding, ensure you're using **Python 3.11** to avoid compatibility issues.

1. **Install Open WebUI**:
   Open your terminal and run the following command to install Open WebUI:

   ```bash
   pip install open-webui
   ```

2. **Running Open WebUI**:
   After installation, you can start Open WebUI by executing:

   ```bash
   open-webui serve
   ```

This will start the Open WebUI server, which you can access at [http://localhost:8080](http://localhost:8080)

### Quick Start with Docker 🐳

> [!NOTE]  
> Please note that for certain Docker environments, additional configurations might be needed. If you encounter any connection issues, our detailed guide on [Open WebUI Documentation](https://docs.openwebui.com/) is ready to assist you.

> [!WARNING]
> When using Docker to install Open WebUI, make sure to include the `-v open-webui:/app/backend/data` in your Docker command. This step is crucial as it ensures your database is properly mounted and prevents any loss of data.

> [!TIP]  
> If you wish to utilize Open WebUI with Ollama included or CUDA acceleration, we recommend utilizing our official images tagged with either `:cuda` or `:ollama`. To enable CUDA, you must install the [Nvidia CUDA container toolkit](https://docs.nvidia.com/dgx/nvidia-container-runtime-upgrade/) on your Linux/WSL system.

#### Installation with Default Configuration

- **If Ollama is on your computer**, use this command:

  ```bash
  docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
  ```

- **If Ollama is on a Different Server**, use this command:

  To connect to Ollama on another server, change the `OLLAMA_BASE_URL` to the server's URL:

  ```bash
  docker run -d -p 3000:8080 -e OLLAMA_BASE_URL=https://example.com -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
  ```

- **To run Open WebUI with Nvidia GPU support**, use this command:

  ```bash
  docker run -d -p 3000:8080 --gpus all --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:cuda
  ```

#### Installation for OpenAI API Usage Only

- **If you're only using OpenAI API**, use this command:

  ```bash
  docker run -d -p 3000:8080 -e OPENAI_API_KEY=your_secret_key -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
  ```

#### Installing Open WebUI with Bundled Ollama Support

This installation method uses a single container image that bundles Open WebUI with Ollama, allowing for a streamlined setup via a single command. Choose the appropriate command based on your hardware setup:

- **With GPU Support**:
  Utilize GPU resources by running the following command:

  ```bash
  docker run -d -p 3000:8080 --gpus=all -v ollama:/root/.ollama -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:ollama
  ```

- **For CPU Only**:
  If you're not using a GPU, use this command instead:

  ```bash
  docker run -d -p 3000:8080 -v ollama:/root/.ollama -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:ollama
  ```

Both commands facilitate a built-in, hassle-free installation of both Open WebUI and Ollama, ensuring that you can get everything up and running swiftly.

After installation, you can access Open WebUI at [http://localhost:3000](http://localhost:3000). Enjoy! 😄

### Other Installation Methods

We offer various installation alternatives, including non-Docker native installation methods, Docker Compose, Kustomize, and Helm. Visit our [Open WebUI Documentation](https://docs.openwebui.com/getting-started/) or join our [Discord community](https://discord.gg/5rJgQTnV4s) for comprehensive guidance.

### Troubleshooting

Encountering connection issues? Our [Open WebUI Documentation](https://docs.openwebui.com/troubleshooting/) has got you covered. For further assistance and to join our vibrant community, visit the [Open WebUI Discord](https://discord.gg/5rJgQTnV4s).

#### Open WebUI: Server Connection Error

If you're experiencing connection issues, it's often due to the WebUI docker container not being able to reach the Ollama server at 127.0.0.1:11434 (host.docker.internal:11434) inside the container. Use the `--network=host` flag in your docker command to resolve this. Note that the port changes from 3000 to 8080, resulting in the link: `http://localhost:8080`.

**Example Docker Command**:

```bash
docker run -d --network=host -v open-webui:/app/backend/data -e OLLAMA_BASE_URL=http://127.0.0.1:11434 --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```

### Keeping Your Docker Installation Up-to-Date

In case you want to update your local Docker installation to the latest version, you can do it with [Watchtower](https://containrrr.dev/watchtower/):

```bash
docker run --rm --volume /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower --run-once open-webui
```

In the last part of the command, replace `open-webui` with your container name if it is different.

Check our Migration Guide available in our [Open WebUI Documentation](https://docs.openwebui.com/tutorials/migration/).

### Ollama Docker image

#### CPU only

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

#### Nvidia GPU
Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation).

##### Install with Apt
1.  Configure the repository
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
```
2.  Install the NVIDIA Container Toolkit packages
```bash
sudo apt-get install -y nvidia-container-toolkit
```

##### Install with Yum or Dnf
1.  Configure the repository

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
    | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
```

2. Install the NVIDIA Container Toolkit packages

```bash
sudo yum install -y nvidia-container-toolkit
```

##### Configure Docker to use Nvidia driver
```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

##### Start the container

```bash
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

#### AMD GPU

To run Ollama using Docker with AMD GPUs, use the `rocm` tag and the following command:

```
docker run -d --device /dev/kfd --device /dev/dri -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama:rocm
```

#### Run model locally

Now you can run a model:

```
docker exec -it ollama ollama run nemotron:70b-instruct-fp16
```

#### Try different models

More models can be found on the [Ollama library](https://ollama.com/library).

## 5. Usage

To launch Rachel HR Interview Bot:

1. Activate the virtual environment (if not already activated). to run this Demo project in Gradio and connect your API (https://build.nvidia.com/nvidia/llama-3_1-nemotron-70b-instruct)
2. Run the main script:
   ```
   pip install nemo-guardrails tensorrt-llm nvidia-nim gradio spacy pytextrank scikit-learn openai
   python -m spacy download en_core_web_sm
   python Rachel.py
   ```
3. Open the provided URL in your web browser to access the Gradio interface.

Using the interface:
1. Upload your resume (PDF format) using the file input.
2. Select your job role from the dropdown menu.
3. Enter the job description in the provided text area.
4. Click "Generate Questions" to start the interview simulation.
5. Interact with Rachel by typing your answers in the chat input.
6. Use the "Skip" button to move to the next question if needed.
7. Click "Generate Answer" to see an expected answer for reference.
8. After completing the interview, click "Provide Feedback" for a comprehensive evaluation.

## 6. Technical Details

### 6.1 Resume Analysis

The resume analysis function uses PyPDF2 to extract text from uploaded PDF files:

```python
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() + '\n'
        return text.strip()
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"
```

This function reads each page of the PDF and concatenates the extracted text, providing a clean string representation of the resume content.

### 6.2 Domain Analysis

The `analyze_domain` function identifies the candidate's specialization based on keywords in the resume:

```python
def analyze_domain(resume_text):
    for domain in job_roles:
        if domain.lower() in resume_text.lower():
            return domain
    return "General"
```

This simple yet effective approach matches resume content against predefined domains, allowing for accurate specialization detection.

## 7. 🤝 Contributing

We welcome contributions to Rachel HR Interview Bot! If you'd like to contribute, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

Please ensure your code adheres to the project's coding standards and include appropriate tests for new features.

## 8. 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 9. Acknowledgments

- Aravindan M (URK22AI1026) for their innovative development of Rachel HR Interview Bot.
- Karunya Institute of Technology and Sciences for supporting this project.
- NVIDIA for providing advanced AI technologies
- OpenWebUI community for the interface framework
- The open-source community for providing the foundational libraries and models used in this project.

---
Built with ❤️ by ARAVINDAN for the NVIDIA and LlamaIndex Developer Contest

Rachel HR Interview Bot represents a significant advancement in AI-assisted interview preparation. By combining cutting-edge NLP techniques, GPU acceleration, and a user-friendly interface, Rachel offers a comprehensive solution for candidates looking to excel in technical HR interviews. We hope this tool proves invaluable in your career journey!
