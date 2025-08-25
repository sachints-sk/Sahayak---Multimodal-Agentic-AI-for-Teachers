import asyncio
import os
import uuid
import json
import re
import traceback
# For Deploy
import vertexai
from vertexai import agent_engines
from dotenv import load_dotenv

# For Agents
from google.adk import Agent
from google.adk.tools import FunctionTool
from google.adk.agents import SequentialAgent
from google.genai import types
from google.adk.tools import VertexAiSearchTool
from vertexai.preview.vision_models import ImageGenerationModel


from difflib import SequenceMatcher
from google.cloud import speech

from google.cloud import texttospeech, storage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# --- Configuration Constants ---
PROJECT_ID = "###################" 
LOCATION = "###################"
COLLECTION_ID = "###################"
GEMINI_2_FLASH = "gemini-2.5-flash"
GEMINI_2_PRO = "gemini-2.5-pro"
AUDIO_BUCKET_NAME = "###################"
OUTPUT_BUCKET_NAME = "###################"
APP_NAME="Agent App"

# For Deploy-----------------------------------------------------
load_dotenv()

vertexai.init(
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION"),
    staging_bucket="###################",
)
# --------------------------------------------------------------
# VisualAidAgent Tool Function
def generate_visual_aid(prompt: str) -> str:
    """
    Generates a simple line drawing or chart based on a teacher's description,
    saves it to Google Cloud Storage, and returns its public URL.

    Args:
        prompt: A description of the visual aid to generate.

    Returns:
        A string containing the success message and the public URL of the generated image.
    """
    print(f"Tool called: Generating visual aid for prompt: '{prompt}'")
    try:
        # Using the exact model name you tested successfully
        model = ImageGenerationModel.from_pretrained("imagen-4.0-ultra-generate-preview-06-06")

        # Using the prompt structure from your successful test
        images = model.generate_images(
            prompt=(
                "simple black and white line drawing, "
                "minimalist, clear outlines, no shading, "
                "suitable for a classroom blackboard, easily reproducible by hand. "
                f"Subject: {prompt}"
            ),
            number_of_images=1,
        )
        image = images[0]
        print("Image generated successfully by the model.")

        image_filename = f"visual-aid-{uuid.uuid4()}.png"
        local_image_path = f"/tmp/{image_filename}"
        
        image.save(location=local_image_path, include_generation_parameters=True)
        print(f"Image saved locally to: {local_image_path}")

        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(OUTPUT_BUCKET_NAME)
        blob = bucket.blob(image_filename)
        
        blob.upload_from_filename(local_image_path)
        print(f"Successfully uploaded image to GCS bucket '{OUTPUT_BUCKET_NAME}'.")

        os.remove(local_image_path)
        print(f"Removed temporary local file: {local_image_path}")

        public_url = f"https://storage.googleapis.com/{OUTPUT_BUCKET_NAME}/{image_filename}"
        print(f"Image is publicly accessible at: {public_url}")

        return f"I have created a visual aid for you. You can view it here: {public_url}"

    except Exception as e:
        print(f"\n--- ERROR IN generate_visual_aid ---")
        traceback.print_exc()
        return f"I'm sorry, I encountered an error while creating the visual aid: {e}"


VisualAidTool = FunctionTool(func=generate_visual_aid)

VisualAidAgent = Agent(
    name="VisualAidAgent",
    model="gemini-2.5-pro",
    tools=[VisualAidTool],
    instruction="""
    **YOUR ROLE:**
    You are a helpful assistant who creates simple visual aids for teachers. You specialize in generating clear, black and white line drawings that are easy to copy onto a blackboard.

    **YOUR PROCESS:**
    1.  Take the teacher's description of the desired visual aid (e.g., "a simple diagram of the water cycle," "a chart showing the parts of a plant").
    2.  You MUST IMMEDIATELY call the `generate_visual_aid` tool.
    3.  Pass the teacher's description directly to the `prompt` parameter of the tool.
    4.  Your ONLY output should be the final response from the tool, which contains the link to the generated image.
    """,
    description="A specialist agent that generates simple line drawings or charts for a blackboard based on a teacher's description.",
)

VisualAidAgentRouter = SequentialAgent(
    name="VisualAidAgentRouter",
    description="Routes requests for creating visual aids, simple drawings, or charts.",
    sub_agents=[
        VisualAidAgent
    ],
)



LessonPlannerAgent = Agent(
    name="LessonPlannerAgent",
    model="gemini-2.5-pro", # Using Pro for higher quality, structured, and pedagogically sound plans
    tools=[], # This agent generates text content and does not require external tools
    instruction="""
    **YOUR ROLE:**
    You are an expert curriculum planner and veteran teacher. You specialize in creating practical, well-structured weekly lesson plans for multi-grade, low-resource classrooms.

    **YOUR GOAL:**
    To take a topic, grade level, and subject from the teacher and create a comprehensive 5-day lesson plan that is easy to follow and implement.

    **YOUR PROCESS:**
    1.  Carefully analyze the user's request to identify the core **topic**, **subject**, and **grade level**.
    2.  Structure your output as a complete 5-day plan. Begin with a clear title, like "**Weekly Lesson Plan: [Subject] - [Topic]**".
    3.  For EACH of the 5 days, you MUST create a section with a clear daily heading (e.g., "**Day 1: Introduction to [Sub-Topic]**").
    4.  Within each day's section, you MUST include and label the following components:
        *   **Objective:** A single, clear sentence stating what students will be able to do by the end of the lesson.
        *   **Opening Activity (5-10 mins):** A brief, engaging warm-up to capture student interest.
        *   **Main Activity (20-25 mins):** The core lesson. Suggest a simple, interactive activity that requires minimal resources (e.g., group discussion, blackboard drawing, a simple experiment with common objects).
        *   **Assessment / Check for Understanding (5 mins):** Suggest a quick, informal way for the teacher to see if students understood the concept (e.g., "Ask three students to share one thing they learned," "Thumbs up/down on a key question").
        *   **Homework (Optional):** If appropriate, suggest a simple, short task.
    5.  Make reasonable assumptions if the user's request is brief. Focus on creating a practical, actionable plan.
    6.  You MUST generate the entire 5-day plan in a single, complete response. Do not ask clarifying questions.
    """,
    description="A specialist agent that creates a complete, 5-day, low-resource lesson plan for any given subject and topic.",
)

LessonPlannerAgentRouter = SequentialAgent(
    name="LessonPlannerAgentRouter",
    description="Routes requests for creating weekly lesson plans, structuring a week of activities, or planning a curriculum for a topic.",
    sub_agents=[
        LessonPlannerAgent
    ],
)



GameGeneratorAgent = Agent(
    name="GameGeneratorAgent",
    model=GEMINI_2_PRO, # Using Pro for more creative and structured game design
    tools=[], # No special tools are needed to generate text-based game rules
    instruction="""
    **YOUR ROLE:**
    You are an expert in educational game design, specializing in creating simple, engaging, and fun classroom games that require minimal resources (like a blackboard, chalk, or just student participation).

    **YOUR GOAL:**
    To transform any educational topic provided by the teacher into a ready-to-play classroom game.

    **YOUR PROCESS:**
    1.  Analyze the user's request to identify the core topic (e.g., fractions, plant types, historical figures), the number of students, and the grade level.
    2.  Invent or adapt a suitable game. It should be simple to explain and quick to set up.
    3.  Generate a complete, well-structured plan for the game. Your response MUST include the following sections, clearly marked:
        *   **Game Name:** A fun and relevant title for the game.
        *   **Learning Objective:** A single sentence explaining what the students will learn or practice.
        *   **Materials Needed:** List only simple items (e.g., "Blackboard and chalk," "Nothing," "Small stones or counters").
        *   **How to Play:** Provide clear, step-by-step instructions for the teacher on how to run the game from start to finish.
        *   **Game Content (if applicable):** Provide a list of 10-15 words, questions, or problems that the teacher can use immediately for the game (e.g., a list of vocabulary words for Pictionary, simple math problems, etc.).
    4.  You MUST provide the complete game plan in a single, final response. Do not ask for more information. Make reasonable assumptions based on the topic.
    """,
    description="A specialist agent that designs simple, fun, and low-resource educational games for the classroom based on a given topic.",
)

GameGeneratorAgentRouter = SequentialAgent(
    name="GameGeneratorAgentRouter",
    description="Routes requests for creating educational classroom games, activities, or fun learning exercises.",
    sub_agents=[
        GameGeneratorAgent
    ],
)

InstantKnowledgeAgent = Agent(
    name="InstantKnowledgeAgent",
    model=GEMINI_2_FLASH, # Flash is suitable for quick, factual-style explanations
    tools=[], # No special tools needed for this agent
    instruction="""
    **YOUR ROLE:**
    You are an expert science and general knowledge communicator for young students. You are an Instant Knowledge Base.

    **YOUR GOAL:**
    Your primary goal is to explain complex topics in a simple, accurate, and engaging way, using easy-to-understand analogies, and delivered in the requested local language.

    **YOUR PROCESS:**
    1.  Carefully analyze the user's query to identify the core complex question (e.g., "Why is the sky blue?", "How do volcanoes erupt?").
    2.  Identify the requested local language for the explanation (e.g., Hindi, Marathi, Tamil).
    3.  Internally, formulate a simple, direct, and accurate answer to the question.
    4.  Next, create a very simple analogy or a real-world example that a child can easily relate to, which helps explain the concept.
    5.  Combine the direct answer and the analogy into a single, cohesive, and friendly explanation.
    6.  Ensure the entire final response is in the requested local language.
    7.  You must provide ONLY the final explanation. Do not ask follow-up questions. Do not offer to create audio. Just provide the text explanation.
    """,
    description="An expert at explaining complex topics (like 'Why is the sky blue?') simply, with analogies, in a specified local language.",
)

InstantKnowledgeAgentRouter = SequentialAgent(
    name="InstantKnowledgeAgentRouter",
    description="Routes requests for simple explanations of complex student questions (e.g., 'why is the sky blue?', 'how do magnets work?').",
    sub_agents=[
        InstantKnowledgeAgent
    ],
)


# In your main agent file, replace the old tool function with this one.

def assess_reading_fluency(original_text: str, student_audio_gcs_uri: str, language_code: str) -> str:
    """
    Analyzes a student's reading audio against an original text to assess fluency.
    Assumes the audio is a WAV file (LINEAR16) with a 16000 Hz sample rate.

    Args:
        original_text: The correct text passage the student was supposed to read.
        student_audio_gcs_uri: The GCS URI of the student's audio file (e.g., 'gs://bucket_name/audio.wav').
        language_code: The BCP-47 language code of the reading passage (e.g., 'en-IN', 'hi-IN').

    Returns:
        A JSON string containing the structured reading assessment report.
    """
    print(f"Tool called: assess_reading_fluency for audio at {student_audio_gcs_uri}")
    try:
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(uri=student_audio_gcs_uri)

        # Correct, simplified configuration for WAV files.
        # It explicitly states the required sample rate.
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,  # CRITICAL: Assuming 16kHz sample rate.
            language_code=language_code,
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
        )

        print("Requesting transcription with explicit sample rate: 16000 Hz...")
        operation = client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=300)

        if not response.results:
             return json.dumps({"error": "Could not understand any speech. The audio file might be silent or have an incorrect sample rate (must be 16000 Hz)."})

        transcript = ""
        words_info = []
        for result in response.results:
            if result.alternatives:
                transcript += result.alternatives[0].transcript + " "
                words_info.extend(result.alternatives[0].words)
        
        transcript = transcript.strip()
        if not transcript:
             return json.dumps({"error": "Transcription was empty. The audio might be silent."})
        print(f"Transcription successful: '{transcript}'")

        # The rest of the function (parsing, metrics, etc.) remains the same.
        def normalize_text(text):
            return re.sub(r'[^\w\s]', '', text.lower())
        original_words = normalize_text(original_text).split()
        transcript_words = normalize_text(transcript).split()
        matcher = SequenceMatcher(None, original_words, transcript_words)
        opcodes = matcher.get_opcodes()
        correct_count=0; skipped_words=[]; added_words=[]; mispronounced_details=[]
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal': correct_count += (i2 - i1)
            elif tag == 'replace': mispronounced_details.append({"expected": " ".join(original_words[i1:i2]), "heard": " ".join(transcript_words[j1:j2])})
            elif tag == 'delete': skipped_words.extend(original_words[i1:i2])
            elif tag == 'insert': added_words.extend(transcript_words[j1:j2])
        total_original_words = len(original_words)
        accuracy = (correct_count / total_original_words) * 100 if total_original_words > 0 else 0
        audio_duration_seconds=0
        if words_info: audio_duration_seconds = words_info[-1].end_time.total_seconds()
        audio_duration_minutes = audio_duration_seconds / 60.0
        wpm = (len(transcript_words) / audio_duration_minutes) if audio_duration_minutes > 0 else 0
        report = {"objective_metrics": {"accuracy_percent": round(accuracy, 1), "words_per_minute": int(wpm), "correct_words": correct_count, "total_words": total_original_words, "audio_duration_seconds": round(audio_duration_seconds, 2)}, "error_analysis": {"mispronounced": mispronounced_details, "skipped": skipped_words, "added": added_words}, "full_transcript": transcript}
        return json.dumps(report)

    except Exception as e:
        print(f"\n--- ERROR IN assess_reading_fluency ---")
        
        # Make the error message helpful for the teacher
        if "sample rate" in str(e):
            return json.dumps({"error": "The audio assessment failed. Please ensure the audio is a WAV file with a 16000 Hz sample rate."})
        return json.dumps({"error": f"An unexpected error occurred during the assessment: {str(e)}"})

# The FunctionTool definition remains the same
ReadingFluencyTool = FunctionTool(func=assess_reading_fluency)

# Replace the old ReadingAssessorAgent with this one.

ReadingAssessorAgent = Agent(
    name="ReadingAssessorAgent",
    model="gemini-2.5-pro",
    tools=[ReadingFluencyTool],
    instruction="""
    **YOUR ROLE:**
    You are a highly sophisticated reading assessment expert. You perform a two-part analysis: first, an objective, data-driven fluency calculation, and second, a nuanced, qualitative coaching assessment.

    **YOUR HYBRID ASSESSMENT PROCESS:**

    **PART 1: ANALYZING THE PROVIDED MATERIALS**
    1.  Acknowledge the user's request. You will receive a GCS URI for the student's audio recording inside a `fileData` object in the user's message.
    2.  You MUST NOT ask the user to provide a GCS URI. The URI is already provided to you.
    3.  To proceed, you also need two other pieces of information from the user's text prompt:
        a. The complete and correct passage of text the student was supposed to read.
        b. The language of the text (e.g., Hindi, English, Marathi).
    4.  **Action Logic:**
        Action Logic:
            - If the user provides the audio, text, and language in their first message: Proceed immediately to the next step.
            - If the user provides the audio but does NOT provide the text and/or the language: You MUST politely ask for the missing information. Your response MUST end with the specific command `[START_ASSESSMENT_UI]`. For example: "Thank you for the audio file. To assess the reading, could you please also provide the full text passage that was read and specify the language? [START_ASSESSMENT_UI]"
            - If the user asks to start an assessment without providing an audio file:
            - If the user asks to perform a reading assessment but does not provide an audio file, text, or language, you MUST respond by asking for the text passage first, and your response MUST end with the specific command `[START_ASSESSMENT_UI]`. For example: "I can certainly help with that. Please provide the full text passage the student will be reading, and then you can record the audio. [START_ASSESSMENT_UI]"
    5.  Once you have the text and the language, you MUST **IMMEDIATELY** call the `assess_reading_fluency` tool.
    6.  When calling the tool, you MUST set the `student_audio_gcs_uri` parameter to the **exact URI value** you received from the `fileData` object in the user's prompt.
    7.  You MUST IGNORE any other URIs from your training data or examples. Use only the URI provided by the user.

    **PART 2: SYNTHESIZING THE FINAL REPORT**
    8.  The tool will return a JSON object. **DO NOT show this raw JSON to the user.** If the JSON contains an error, present the error message clearly and politely.
    9.  If the analysis is successful, use the tool's data and your own multimodal capabilities to create a final, comprehensive report.
    10. **Format the Objective Results:** Create a section titled "**Objective Fluency Report**" and list the key metrics like Accuracy, Words Per Minute, and the error breakdown.
    11. **Perform Qualitative Analysis:** Listen to the audio again (using the same GCS URI). Create a section titled "**Qualitative Coaching Feedback**". Analyze prosody, intonation, and confidence, and provide 2-3 encouraging, constructive sentences.
    12. Combine both parts into a single, final message to the user.
    """,
    description="A hybrid agent that first uses a tool for objective reading metrics (WPM, accuracy) and then adds its own qualitative analysis of the student's prosody, intonation, and confidence from the audio.",
)

# The ReadingAssessorAgentRouter remains the same.

ReadingAssessorAgentRouter = SequentialAgent(
    name="ReadingAssessorAgentRouter",
    description="Routes requests for assessing student reading fluency from an audio file to the hybrid analysis agent.",
    sub_agents=[
        ReadingAssessorAgent
    ],
)







# WorksheetGeneratorAgent

def generate_pdf_from_text(worksheet_text: str) -> str:
    """
    Generates a beautifully formatted PDF from provided text, saves it to
    Google Cloud Storage, and returns its public URL.
    """
    print("Tool called: Generating enhanced PDF from worksheet text.")
    try:
        storage_client = storage.Client(project=PROJECT_ID)

        pdf_filename = f"{uuid.uuid4()}.pdf"
        # Use the /tmp/ directory for temporary storage, which is standard for cloud environments
        local_pdf_path = f"/tmp/{pdf_filename}"

        print(f"Preparing to create PDF at: {local_pdf_path}")

        doc = SimpleDocTemplate(local_pdf_path, pagesize=letter,
                                rightMargin=0.75*inch, leftMargin=0.75*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)

        # Define Custom Styles
        styles = getSampleStyleSheet()
        # (All your custom style definitions go here)
        styles.add(ParagraphStyle(name='TitleStyle', fontName='Helvetica-Bold', fontSize=18, leading=22, spaceAfter=14, textColor=colors.HexColor("#4A90E2"), alignment=1))
        styles.add(ParagraphStyle(name='ActivityTitle', fontName='Helvetica-Bold', fontSize=14, leading=18, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#333333")))
        styles.add(ParagraphStyle(name='InstructionStyle', fontName='Helvetica-Oblique', fontSize=10, leading=12, spaceAfter=6, textColor=colors.darkgray))
        styles.add(ParagraphStyle(name='BodyStyle', fontName='Helvetica', fontSize=11, leading=14, spaceAfter=6, wordWrap='CJK'))
        styles.add(ParagraphStyle(name='HeaderStyle', fontName='Helvetica', fontSize=11, leading=14, spaceAfter=0))
        styles.add(ParagraphStyle(name='MonospaceStyle', fontName='Courier', fontSize=10, leading=12, spaceAfter=6, textColor=colors.darkgrey))


        story = []
        lines = worksheet_text.strip().split('\n')
        
        # Helper to process inline bolding (**)
        def format_bold(text):
            parts = text.split('**')
            result = []
            for i, part in enumerate(parts):
                result.append(f"<b>{part}</b>" if i % 2 == 1 else part)
            return "".join(result)

        # Intelligent Line-by-Line Parsing
        for line in lines:
            line_content = format_bold(line)
            if line.strip().startswith('+--'):
                 story.append(Paragraph(line_content, styles['MonospaceStyle']))
            elif line.strip().startswith('**Activity'):
                story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey, spaceAfter=5))
                story.append(Paragraph(line_content, styles['ActivityTitle']))
            elif line.strip().startswith('**'):
                 story.append(Paragraph(line_content, styles['TitleStyle']))
            elif "Name:" in line or "Date:" in line:
                story.append(Paragraph(line_content, styles['HeaderStyle']))
                if "Date:" in line: story.append(Spacer(1, 0.25*inch))
            elif line.strip().startswith('*'):
                 story.append(Paragraph(line_content.replace('*','<i>',1).replace('*','</i>',1), styles['InstructionStyle']))
            elif "Draw it in the box below!" in line:
                story.append(Paragraph(line_content, styles['InstructionStyle']))
                story.append(Spacer(1, 0.2*inch))
                story.append(HRFlowable(width="80%", thickness=1, color=colors.black, hAlign='CENTER'))
                story.append(Spacer(1, 2.5*inch))
                story.append(HRFlowable(width="80%", thickness=1, color=colors.black, hAlign='CENTER'))
            elif not line.strip():
                story.append(Spacer(1, 0.1*inch))
            else:
                story.append(Paragraph(line_content, styles['BodyStyle']))

        doc.build(story)
        print(f"PDF successfully created locally at: {local_pdf_path}")

        # Upload to GCS
        bucket = storage_client.bucket(OUTPUT_BUCKET_NAME)
        blob = bucket.blob(pdf_filename)
        blob.upload_from_filename(local_pdf_path)
        print(f"Successfully uploaded to GCS.")

        os.remove(local_pdf_path)
        public_url = f"https://storage.googleapis.com/{OUTPUT_BUCKET_NAME}/{pdf_filename}"
        print(f"PDF is public at: {public_url}")

        return f"The worksheet PDF has been successfully created. You can download it here: {public_url}"

    except Exception as e:
        print(f"\n--- ERROR IN generate_pdf_from_text ---")
        traceback.print_exc()
        return "I'm sorry, I encountered an error creating the PDF."


WorksheetToPdfTool = FunctionTool(func=generate_pdf_from_text)

# WorksheetGeneratorAgent (UPDATED with new instructions)
WorksheetGeneratorAgent = Agent(
    name="WorksheetGeneratorAgent",
    model="gemini-2.5-pro", 
    tools=[WorksheetToPdfTool],
    instruction="""
**YOUR ROLE:**
    You are an expert curriculum designer. Your specialty is creating beautiful, functional, and differentiated worksheets from a textbook page for students at different learning levels. Your goal is to provide a ready-to-print PDF in a single step.

    **YOUR PROCESS:**
    1.  Thoroughly analyze the user's request and the provided image of the textbook page.
    2.  Internally, compose the complete text for the worksheet. You MUST generate this text following our specific markdown format so it can be turned into a beautiful PDF.
    3.  **FORMATTING RULES (MANDATORY):**
        *   The main title MUST start and end with `**` (e.g., `**Worksheet: The Solar System**`).
        *   You MUST include lines for `Name: _________________________` and `Date: __________________________`.
        *   Each activity title MUST start with `**Activity` and end with `**`.
        *   Instructions should be on their own line and start and end with `*` (e.g., `*Circle the correct answer.*`).
        *   Use `____________________` for "fill in the blank" questions.
        *   For a word box, use text art, like `+------------------+`.
        *   For a drawing activity, include the instruction "Draw it in the box below!".
    4.  **CRITICAL STEP:** Once you have composed the complete markdown text in your memory, you MUST **IMMEDIATELY and AUTOMATICALLY** call the `generate_pdf_from_text` tool.
    5.  Pass the entire, perfectly formatted markdown text you just created directly to the tool.
    6.  **DO NOT** show the markdown text to the user.
    7.  **DO NOT** ask any follow-up questions.
    8.  Your **ONLY** output should be the final response from the tool, which contains the link to the generated PDF. For example: "I have created your printable worksheet. You can download it here: [URL]".
    """,
    description="A specialist agent that takes a photo of a textbook page and directly generates a link to a printable PDF worksheet.",
)

WorksheetGeneratorAgentRouter = SequentialAgent(
    name="WorksheetGeneratorAgentRouter",
    description="Routes requests for creating worksheets from textbook pages.",
    sub_agents=[
        WorksheetGeneratorAgent
    ],
)



# HyperLocalContentAgent

def generate_audio_from_text(text: str, language_code: str, voice_name: str) -> str:
    """
    Generates an audio file from the provided text and returns its public URL. This tool
    creates a .wav file with LINEAR16 encoding and returns a public GCS URL.

    Args:
        text: The text to be converted into speech.
        language_code: The BCP-47 language code for the text (e.g., 'mr-IN' for Marathi).
        voice_name: The specific voice to use for synthesis (e.g., 'mr-IN-Wavenet-A').

    Returns:
        A string containing the success message and the public URL of the generated .wav audio file.
    """
    print(f"Tool called: Generating audio for language '{language_code}'.")
    try:
        # Use the correct client for long audio synthesis
        tts_client = texttospeech.TextToSpeechLongAudioSynthesizeClient()
        # Explicitly set the project to avoid authentication context issues
        storage_client = storage.Client(project=PROJECT_ID)

        # Use .wav extension to match the LINEAR16 encoding
        output_filename = f"{uuid.uuid4()}.wav"
        gcs_output_uri = f"gs://{AUDIO_BUCKET_NAME}/{output_filename}"

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code, name=voice_name
        )
        
        # Long Audio Synthesis requires LINEAR16 encoding
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )
        
        request = texttospeech.SynthesizeLongAudioRequest(
            parent=f"projects/{PROJECT_ID}/locations/{LOCATION}",
            input=synthesis_input,
            audio_config=audio_config,
            voice=voice,
            output_gcs_uri=gcs_output_uri,
        )

        operation = tts_client.synthesize_long_audio(request=request)
        
        print("Waiting for audio synthesis operation to complete...")
        operation.result(timeout=300) 
        print("Synthesis complete.")

        # Manually construct the public URL for the file
        # This is for buckets with uniform public access (no need for make_public())
        public_url = f"https://storage.googleapis.com/{AUDIO_BUCKET_NAME}/{output_filename}"
        print(f"Audio file is public at: {public_url}")
        
        return f"The audio has been successfully created. You can listen to it here: {public_url}"

    except Exception as e:
        # Log the detailed error to the server console for future debugging
        print("\n--- ERROR IN generate_audio_from_text ---")
       
        print("--- END OF ERROR ---\n")
        return "I'm sorry, I encountered an error while trying to create the audio file."


TextToSpeechTool = FunctionTool(func=generate_audio_from_text)



HyperLocalContentAgent = Agent(
    name="HyperLocalContentAgent",
    model=GEMINI_2_FLASH, 
    tools=[TextToSpeechTool], 
    instruction="""
    **YOUR ROLE:**
    You are a creative storyteller and a cultural expert. Your purpose is to generate highly engaging, culturally relevant, and educational content in various local Indian languages. You can also turn your stories into audio.

    **YOUR PROCESS:**
    1.  Analyze the user's request to identify the target language (e.g., Marathi, Hindi, Tamil, etc.) and the core educational concept.
    2.  If you can fulfill the request, create the requested content (story, analogy, etc.) entirely in the specified local language.
    3.  Present the generated text to the user.
    4.  **CRITICAL STEP:** After you provide the text, ALWAYS ask a follow-up question in English: "Would you like an audio version of this story?"
    5.  **If the user says 'yes' or agrees:**
        - You MUST use the `generate_audio_from_text` tool.
        - You must find the correct `language_code` and `voice_name` from the **Voice Selection Rules** below.
        - Pass the full text of the story you just generated to the tool.
        - When the tool returns a file path, format your response like this example:'I've converted your text to speech. The audio file is saved at `/path/to/file.mp3`
        - Make sure to put ONLY the file path inside backticks (`), not any additional text
        - Never modify or abbreviate the path
        - This exact format is critical for proper processing.
    6.  **If a requested language is NOT in the rules below**, you must state that you cannot create audio for that language yet and ask if they would like the story in a different supported language.

    **VOICE SELECTION RULES:**
    - **Bengali:** Use `language_code='bn-IN'` and `voice_name='bn-IN-Chirp3-HD-Callirrhoe'`.
    - **English (India):** Use `language_code='en-IN'` and `voice_name='en-IN-Chirp3-HD-Callirrhoe'`.
    - **Gujarati:** Use `language_code='gu-IN'` and `voice_name='gu-IN-Chirp3-HD-Callirrhoe'`.
    - **Hindi:** Use `language_code='hi-IN'` and `voice_name='hi-IN-Chirp3-HD-Callirrhoe'`.
    - **Kannada:** Use `language_code='kn-IN'` and `voice_name='kn-IN-Chirp3-HD-Callirrhoe'`.
    - **Malayalam:** Use `language_code='ml-IN'` and `voice_name='ml-IN-Chirp3-HD-Callirrhoe'`.
    - **Marathi:** Use `language_code='mr-IN'` and `voice_name='mr-IN-Chirp3-HD-Callirrhoe'`.
    - **Punjabi:** Use `language_code='pa-IN'` and `voice_name='pa-IN-Chirp3-HD-Callirrhoe'`.
    - **Tamil:** Use `language_code='ta-IN'` and `voice_name='ta-IN-Chirp3-HD-Callirrhoe'`.
    - **Telugu:** Use `language_code='te-IN'` and `voice_name='te-IN-Chirp3-HD-Callirrhoe'`.
    - **Urdu:** Use `language_code='ur-IN'` and `voice_name='ur-IN-Chirp3-HD-Callirrhoe'`.

    """,
    description="A specialist agent that generates culturally relevant content in local languages and can create audio versions of it.",
)

HyperLocalContentAgentRouter = SequentialAgent(
    name="HyperLocalContentAgentRouter",
    description="Routes requests for creating stories, examples, or analogies in local languages.",
    sub_agents=[
        HyperLocalContentAgent
    ],
)

# NCERTKnowledgeBaseAgent

class_1_hindi_id = "projects/###################/locations/global/collections/default_collection/dataStores/class-1-subject-hindi"
class_1_english_id = "projects/###################/locations/global/collections/default_collection/dataStores/class-1-subject-english"
class_1_maths_id = "projects/###################/locations/global/collections/default_collection/dataStores/class-1-subject-maths"
class_2_hindi_id = "projects/###################/locations/global/collections/default_collection/dataStores/class-2-subject-hindi"
class_2_english_id = "projects/###################/locations/global/collections/default_collection/dataStores/class-2-subject-english"
class_2_maths_id = "projects/###################/locations/global/collections/default_collection/dataStores/class-2-subject-maths"


# Tool Instantiation
vertex_search_tool_class_1_hindi = VertexAiSearchTool(
    data_store_id=class_1_hindi_id,
)
vertex_search_tool_class_1_english = VertexAiSearchTool(
    data_store_id=class_1_english_id,    
)
vertex_search_tool_class_1_maths = VertexAiSearchTool(
    data_store_id=class_1_maths_id,    
)
vertex_search_tool_class_2_hindi = VertexAiSearchTool(
    data_store_id=class_2_hindi_id,
)
vertex_search_tool_class_2_english = VertexAiSearchTool(
    data_store_id=class_2_english_id,    
)
vertex_search_tool_class_2_maths = VertexAiSearchTool(
    data_store_id=class_2_maths_id,    
)



NCERTKnowledgeBaseAgent = Agent(
    name="NCERTKnowledgeBaseAgent",
    model=GEMINI_2_FLASH,
    tools=[
        vertex_search_tool_class_1_hindi,
        vertex_search_tool_class_1_english,
        vertex_search_tool_class_1_maths,
        vertex_search_tool_class_2_hindi,
        vertex_search_tool_class_2_english,
        vertex_search_tool_class_2_maths,
        # ... a very long list of tools
    ],
    instruction="""
    **YOUR ROLE:**
    You are an expert researcher specializing in NCERT textbooks. You can find, compare, and synthesize information from multiple books to answer complex questions.

    **YOUR PROCESS:**
    1.  Carefully analyze the user's query to understand exactly what information is needed.
    2.  Identify ALL the textbooks (class number and subject) required to answer the question.
    3.  Analyze the user's query to determine the precise class and subject.
        Then, select the ONE specific tool that matches that class and subject to perform the search.
    4.  **If the query requires information from multiple books (e.g., a comparison), you MUST call the appropriate tool multiple times, once for each book.**  
    5.  Wait until you have gathered ALL the necessary information from ALL your tool calls.
    6.  Finally, synthesize the collected information into a single, comprehensive answer that directly addresses the user's original question.
    7.  If any piece of information cannot be found, state that clearly in your final answer.
    """,
    description="A specialist agent that can find, compare, and synthesize information from any NCERT textbook across all classes and subjects.",
)

NCERTKnowledgeBaseAgentRouter = SequentialAgent(
    name="NCERTKnowledgeBaseAgentRouter",
    description="Routs the user to the appropriate NCERTKnowledgeBaseAgent",
    sub_agents=[
        NCERTKnowledgeBaseAgent
    ],
)


# Root Agent

root_agent = Agent(
    name="Sahayak",
    model=GEMINI_2_FLASH,
    description="You are teaching assistant (\"Sahayak\") that empowers teachers in multi-grade, low-resource environments.",
    instruction="""
    You are "Sahayak," a friendly and highly capable AI teaching assistant. Your purpose is to make a teacher's job easier by providing versatile support.

    **YOUR PRIMARY ROLE:**
    1.  Start by warmly greeting the teacher in a supportive tone.
    2.  Briefly introduce your capabilities. You can help with:
        *   Creating stories and examples in local languages.
        *   Making worksheets from textbook pages for different student levels.
        *   Answering complex student questions simply.
        *   Designing simple drawings for the blackboard.
        *   Building weekly lesson plans.
        *   Finding specific information and fact-checking from official NCERT textbooks.
    3.  Ask the teacher what they need help with today.
    4.  Carefully analyze the teacher's request to determine their main goal.
    5.  Based on their goal, you MUST transfer control to the correct specialist sub-agent. DO NOT try to perform the task yourself.

    **DECISION-MAKING AND ROUTING RULES:**

        - If the teacher wants a story, an example, an analogy, or any creative content in a specific local language (like the user input: "Create a story in Marathi about farmers to explain soil types"), you MUST call the HyperLocalContentAgentRouter.
        - If the teacher wants to find information from NCERT books, fact-check a concept, or asks a question specifically about textbook content (e.g., "what does the class 7 textbook say about the Mughal empire?"), you MUST call the NCERTKnowledgeBaseAgentRouter.
        - If the teacher wants to create a worksheet, especially from a textbook page, an image, or a photo, you MUST call the WorksheetGeneratorAgentRouter.
        - If the teacher wants to assess reading, check reading fluency, analyze a student's audio recording of a text, or asks to "check my student's reading", you MUST call the ReadingAssessorAgentRouter.
        - If the teacher asks a "why" or "how" question, or wants a simple explanation for a complex topic (e.g., "Explain why the sky is blue in Hindi"), you MUST call the InstantKnowledgeAgentRouter.
        - If the teacher wants to create a game, a fun activity, or an interactive exercise on a topic (e.g., "Make a game about fractions"), you MUST call the GameGeneratorAgentRouter.
        - If the teacher asks for a lesson plan, a weekly plan, or a curriculum structure for a topic (e.g., "Create a 5-day lesson plan for Class 3 on the water cycle"), you MUST call the LessonPlannerAgentRouter.
        - If the teacher asks for a drawing, a chart, a diagram, or a visual aid for the blackboard (e.g., "draw me the water cycle," "create a simple chart of the plant cell"), you MUST call the VisualAidAgentRouter.
    """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0,
    ),
    
    # Add the new router to the list of sub-agents
    sub_agents=[NCERTKnowledgeBaseAgentRouter, HyperLocalContentAgentRouter, WorksheetGeneratorAgentRouter, ReadingAssessorAgentRouter, InstantKnowledgeAgentRouter, GameGeneratorAgentRouter , LessonPlannerAgentRouter , VisualAidAgentRouter],
)


# For Deploy-----------------------------------------------------
if __name__ == "__main__":

    print(f"Starting deployment of agent '{APP_NAME}'...")
   
    updated_requirements = [
        "google-cloud-aiplatform[adk,agent_engines]>=1.100.0",
        "python-dotenv",
        "google-cloud-texttospeech>=2.27.0",
        "google-cloud-storage",
        "reportlab>=4.4.2",
        "google-cloud-speech>=2.33.0"
    ]

    remote_app = agent_engines.create(
        display_name=APP_NAME,
        agent_engine=root_agent,
        requirements=updated_requirements,
    )

    print(f"Agent deployed successfully: {remote_app.resource_name}") 
