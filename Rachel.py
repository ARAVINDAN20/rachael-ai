# INSTALL all this Libraries("pip install nemo-guardrails tensorrt-llm nvidia-nim gradio spacy pytextrank scikit-learn openai")
# Install this ("python -m spacy download en_core_web_sm")
# RUN "python Rachel.py"
import gradio as gr
import PyPDF2
import spacy
import pytextrank
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import nemo_guardrails as guardrails
from tensorrt_llm import TensorRTLLM
import nvidia.nim as nim
import json
import logging
from typing import List, Dict, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterviewPrepSystem:
    def __init__(self, api_key: str):
        """
        Initialize the Interview Preparation System with all NVIDIA components
        """
        # Initialize OpenAI client with NVIDIA API
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )

        # Initialize NeMo Guardrails
        self.guardrails_config = guardrails.RailsConfig.from_content(
            """
            # Response constraints for the interview system
            define rules:
                - no harmful or inappropriate content
                - maintain professional tone
                - focus on technical accuracy
                - avoid sensitive personal information
                - ensure fair evaluation criteria
            
            define flow interview_question:
                when: user asks for interview questions
                do: generate technical questions
                must: verify questions are relevant and appropriate
            
            define flow answer_evaluation:
                when: evaluating user response
                do: provide constructive feedback
                must: be objective and encouraging
            """
        )
        self.rails = guardrails.RailsExecutor(self.guardrails_config)

        # Initialize NIM client for microservices
        self.nim_client = nim.Client()
        self.model_config = {
            "model_name": "nvidia/llama-3.1-nemotron-70b-instruct",
            "max_tokens": 1024,
            "temperature": 0.7
        }

        # Initialize TensorRT-LLM for optimized inference
        self.trt_llm = TensorRTLLM()
        self.trt_llm.initialize_engine(
            model_name="nvidia/llama-3.1-nemotron-70b-instruct",
            precision="fp16",
            max_batch_size=1
        )

        # Initialize NLP components
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("textrank")
        self.tfidf_vectorizer = TfidfVectorizer()

    async def generate_secure_response(self, prompt: str) -> str:
        """
        Generate a response using NeMo Guardrails and NIM microservices
        """
        try:
            # Apply guardrails to the prompt
            safe_prompt = await self.rails.process_input(prompt)
            
            # Use NIM microservice for inference
            response = await self.nim_client.generate(
                model=self.model_config["model_name"],
                prompt=safe_prompt,
                max_tokens=self.model_config["max_tokens"],
                temperature=self.model_config["temperature"]
            )
            
            # Apply guardrails to the response
            safe_response = await self.rails.process_output(response)
            
            return safe_response
        except Exception as e:
            logger.error(f"Error in generate_secure_response: {str(e)}")
            return "Error generating response. Please try again."

    async def optimize_inference(self, input_text: str) -> str:
        """
        Use TensorRT-LLM for optimized inference
        """
        try:
            optimized_response = await self.trt_llm.generate(
                input_text,
                max_length=1024,
                temperature=0.7
            )
            return optimized_response
        except Exception as e:
            logger.error(f"Error in optimize_inference: {str(e)}")
            return None

    async def generate_hr_questions(self, domain: str, job_role: str, job_description: str) -> List[str]:
        """
        Generate HR questions using optimized inference and guardrails
        """
        prompt = f"""Generate 5 high-quality Technical HR interview questions for a candidate 
        specializing in {domain} for the role of {job_role} with the following job description:
        {job_description}
        Focus on advanced concepts and industry best practices."""

        # Generate questions using optimized inference
        response = await self.optimize_inference(prompt)
        if not response:
            response = await self.generate_secure_response(prompt)

        # Process and validate questions
        questions = [q.strip() for q in response.split('\n') 
                    if q.strip() and not q.strip().startswith(('Question', 'Q:', '#'))]
        return questions[:5]

    async def evaluate_answer(self, 
                            question: str, 
                            user_answer: str, 
                            reference_answer: str) -> Dict:
        """
        Evaluate user answers using advanced NLP and optimized inference
        """
        # Extract keywords using TextRank
        user_keywords = set(self.extract_keywords_textrank(user_answer.lower()))
        reference_keywords = set(self.extract_keywords_textrank(reference_answer.lower()))
        question_keywords = set(self.extract_keywords_textrank(question.lower()))

        # Calculate keyword relevance
        relevant_keywords = question_keywords.intersection(reference_keywords)
        user_relevant_keywords = user_keywords.intersection(relevant_keywords)
        keyword_relevance = len(user_relevant_keywords) / len(relevant_keywords) if relevant_keywords else 0

        # Calculate semantic similarity using TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([user_answer.lower(), reference_answer.lower()])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # Calculate final score
        final_score = (0.6 * keyword_relevance + 0.4 * cosine_sim) * 10
        rating = round(final_score)

        # Generate feedback using optimized inference
        feedback_prompt = f"""Analyze the following interview response:
        Question: {question}
        User Answer: {user_answer}
        Reference Answer: {reference_answer}
        Score: {rating}/10
        
        Provide constructive feedback focusing on:
        1. Strengths
        2. Areas for improvement
        3. Missing key concepts
        4. Suggestions for better answers"""

        feedback = await self.optimize_inference(feedback_prompt)
        if not feedback:
            feedback = await self.generate_secure_response(feedback_prompt)

        return {
            "rating": rating,
            "feedback": feedback,
            "keyword_analysis": {
                "present": list(user_relevant_keywords),
                "missing": list(relevant_keywords - user_relevant_keywords)
            }
        }

    def extract_keywords_textrank(self, text: str) -> List[str]:
        """
        Extract keywords using TextRank algorithm
        """
        doc = self.nlp(text)
        return [phrase.text for phrase in doc._.phrases[:10]]

# GUI Implementation with advanced features
class InterviewPrepGUI:
    def __init__(self, system: InterviewPrepSystem):
        self.system = system
        self.setup_gui()

    def setup_gui(self):
        with gr.Blocks(css=self.get_css()) as self.demo:
            gr.Markdown("# 🎓 Advanced Interview Preparation System")
            
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("Resume Analysis", open=False):
                        self.file_input = gr.File(
                            label="📄 Upload your resume (PDF)", 
                            file_types=['pdf']
                        )
                        self.upload_button = gr.Button("📤 Upload and Analyze Resume")
                        self.upload_status = gr.Textbox(label="Status")
                        self.detected_domain = gr.Textbox(label="🎯 Detected Specialization")
                        self.job_role_dropdown = gr.Dropdown(
                            label="🔍 Select Job Role", 
                            choices=[]
                        )
                        self.job_description_input = gr.Textbox(
                            label="📋 Enter Job Description", 
                            max_lines=10
                        )
                    
                    self.generate_button = gr.Button(
                        "🔄 Generate Questions", 
                        elem_classes=["generate-btn"]
                    )
                    self.feedback_button = gr.Button(
                        "📝 Provide Feedback", 
                        elem_classes=["feedback-btn"]
                    )

                with gr.Column(scale=2):
                    self.chatbot = gr.Chatbot(label="💬 Interview Simulation")
                    self.chat_input = gr.Textbox(
                        label="Your Answer", 
                        placeholder="Type your answer here..."
                    )
                    with gr.Row():
                        self.chat_button = gr.Button("📨 Submit Answer")
                        self.skip_button = gr.Button("⏭️ Skip Question")
                        self.hint_button = gr.Button("💡 Get Hint")

            # State management
            self.chat_history = gr.State([])
            self.questions = gr.State([])
            self.current_question_index = gr.State(0)
            self.user_answers = gr.State([])

            # Event handlers
            self.setup_event_handlers()

    def setup_event_handlers(self):
        self.upload_button.click(
            self.handle_resume_upload,
            inputs=[self.file_input],
            outputs=[self.upload_status, self.detected_domain, self.job_role_dropdown]
        )
        self.generate_button.click(
            self.handle_question_generation,
            inputs=[
                self.detected_domain,
                self.job_role_dropdown,
                self.job_description_input,
                self.chat_history
            ],
            outputs=[
                self.chatbot,
                self.chat_history,
                self.questions,
                self.current_question_index,
                self.user_answers
            ]
        )
        # Add more event handlers...

    @staticmethod
    def get_css() -> str:
        return """
        /* Your existing CSS styles */
        """

# Main application setup
def main():
    # Initialize the system
    system = InterviewPrepSystem(
        api_key="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" #connect YOUR aip (https://build.nvidia.com/nvidia/llama-3_1-nemotron-70b-instruct)
    )
    
    # Initialize and launch the GUI
    gui = InterviewPrepGUI(system)
    gui.demo.launch()

if __name__ == "__main__":
    main()