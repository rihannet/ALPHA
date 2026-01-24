# ALPHA - Autism Detection System

## Overview

ALPHA is an AI-powered autism detection system that uses video analysis and questionnaires to identify autism spectrum disorders and provide personalized therapeutic activities.

## What Does ALPHA Do?

1. **Detects Autism Type**: Analyzes child behavior through video and questions
2. **Provides Activities**: Suggests personalized activities based on autism type
3. **Tracks Progress**: Monitors child's emotional response to activities
4. **Connects to Specialists**: Shows nearby autism specialists

## How It Works

### Simple 5-Step Process

```
1. Sign Up → Create your account
2. Upload Video → Record 30-60 second video of your child
3. Answer Questions → 5 simple questions about behavior
4. Get Results → AI predicts autism type
5. Start Activities → Personalized activities for your child
```

## Installation

### Requirements
- Python 3.8 or higher
- Windows/Mac/Linux
- 4GB RAM minimum
- 2GB free disk space

### Quick Setup

1. **Install Python packages**:
```bash
pip install torch torchvision
pip install transformers
pip install tensorflow
pip install PyQt5
pip install pandas openpyxl
pip install opencv-python
pip install scikit-learn
pip install joblib
```

2. **Download Models**:
- Place `autism_model.pth` in root folder
- Place `resnet3d_model.h5` in root folder
- Ensure `trained_models/` folder exists with all files

3. **Run Application**:
```bash
python aplha_mian.py
```

## Usage Guide

### For New Users

1. **Launch Application**
   - Run `python aplha_mian.py`
   - Click "Sign Up" button

2. **Create Account**
   - Enter email, username, password
   - Click "Sign up"
   - You'll automatically go to video upload

3. **Upload Video** (Stage 1)
   - Record 30-60 second video of your child
   - Show typical behaviors and expressions
   - Click "Upload the video"
   - Wait for analysis (~1 second)
   - Click "NEXT"

4. **Answer Questions** (Stage 2)
   - 5 questions about your child's behavior
   - Type your answers in the text box
   - Click "NEXT" after each answer
   - System analyzes all answers together

5. **View Results**
   - System combines video + questionnaire analysis
   - Shows predicted autism type
   - Automatically saves result

6. **Home Page**
   - Two options:
     - **ACTIVATE MODE**: Start activities
     - **GUIDE MODE**: Find specialists

### For Returning Users

1. **Login**
   - Enter username and password
   - Click "Login"
   - Go directly to Home Page

2. **Choose Mode**
   - **Activate Mode**: Continue with activities
   - **Guide Mode**: View specialist information

## Features

### 1. Video Analysis (CV Model)
- **Technology**: Deep learning (ResNet3D)
- **Input**: 30-60 second video
- **Analysis**: Facial expressions, movements, behaviors
- **Speed**: ~0.5 seconds
- **Accuracy**: 85%

### 2. Questionnaire Analysis (NLP Model)
- **Technology**: BERT AI language model
- **Input**: 5 questions about behavior
- **Analysis**: Communication, social skills, routines
- **Speed**: ~1 second
- **Accuracy**: 98%

### 3. Personalized Activities
- **8 activities** per autism type
- **Visual guides** with images
- **Progress tracking** with emotion detection
- **Automatic advancement** when child is happy

### 4. Emotion Detection
- **Real-time analysis** of child's response
- **3 emotions**: Happy, Sad, Angry
- **Automatic progression** to next activity when happy
- **Confidence threshold**: 80%

### 5. Specialist Finder
- **List of autism specialists**
- **Contact information**
- **Specialties displayed**

## Autism Types Detected

The system can identify 6 types of autism spectrum disorders:

1. **Asperger's Syndrome**
   - High-functioning
   - Good language skills
   - Social challenges

2. **Classic Autism**
   - Moderate to severe
   - Communication difficulties
   - Repetitive behaviors

3. **PDD-NOS** (Pervasive Developmental Disorder - Not Otherwise Specified)
   - Atypical autism
   - Some autism traits
   - Doesn't fit other categories

4. **Rett Syndrome**
   - Rare genetic disorder
   - Affects girls primarily
   - Loss of motor skills

5. **High-Functioning Autism (HFA)**
   - Similar to Asperger's
   - Better outcomes
   - Good cognitive abilities

6. **Childhood Disintegrative Disorder (CDD)**
   - Rare condition
   - Late onset (after age 2)
   - Loss of previously acquired skills

## File Structure

```
ALPHA_FINEL/
├── aplha_mian.py          # Main application (run this!)
├── autism_model.pth        # CV model for video analysis
├── resnet3d_model.h5       # Emotion detection model
├── user.xlsx               # User database
├── alpha_type.txt          # Current user's autism type
├── autism_questions.csv    # Questions for NLP stage
│
├── trained_models/         # NLP model files
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer files
│   └── label_encoder.pkl
│
├── activates/              # Activity images (48 total)
│   ├── Asperger's Syndrome/
│   ├── Classic Autism/
│   ├── PDD-NOS/
│   ├── Rett Syndrome/
│   ├── HFA/
│   └── Childhood Disintegrative Disorder/
│
├── ui imges/               # UI graphics
│   └── background images
│
└── *.ui files              # UI layouts (PyQt5)
```

## Troubleshooting

### Common Issues

**Problem**: "Model file not found"
- **Solution**: Ensure all model files are in the correct location
- Check: `autism_model.pth`, `resnet3d_model.h5`, `trained_models/`

**Problem**: "Video upload failed"
- **Solution**: 
  - Use .mp4 format
  - Ensure video is 30-60 seconds
  - Check video is not corrupted

**Problem**: "Excel file permission denied"
- **Solution**: Close `user.xlsx` if open in Excel
- Check file is not read-only

**Problem**: "CUDA out of memory"
- **Solution**: System will automatically use CPU
- Close other applications to free memory

**Problem**: "Application won't start"
- **Solution**: 
  - Check all Python packages installed
  - Run: `pip install -r requirements.txt`
  - Verify Python version ≥ 3.8

### Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| "Invalid username or password" | Login credentials wrong | Check spelling, try sign up |
| "Username already exists" | Account exists | Use different username or login |
| "Video is too short" | Video < 16 frames | Record longer video (30s+) |
| "No video selected" | No file uploaded | Click upload and select video |
| "Model prediction failed" | Model error | Restart application |

## Tips for Best Results

### Video Recording Tips
1. **Good lighting**: Record in well-lit area
2. **Clear view**: Show child's face clearly
3. **Natural behavior**: Capture typical activities
4. **Duration**: 30-60 seconds is ideal
5. **Stability**: Keep camera steady

### Questionnaire Tips
1. **Be honest**: Answer truthfully
2. **Be specific**: Give detailed examples
3. **Be consistent**: Answer all questions
4. **Take time**: Think before answering
5. **Ask for help**: Consult with caregivers if needed

### Activity Tips
1. **Start simple**: Begin with easier activities
2. **Be patient**: Give child time to engage
3. **Record response**: Upload video after activity
4. **Celebrate success**: Positive reinforcement
5. **Repeat if needed**: Stay on activity until happy

## System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14, Ubuntu 18.04
- **CPU**: Intel i5 or equivalent
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Camera**: For recording videos (optional)

### Recommended Requirements
- **OS**: Windows 11, macOS 12+, Ubuntu 20.04+
- **CPU**: Intel i7 or equivalent
- **RAM**: 8GB
- **GPU**: NVIDIA GPU with CUDA support (optional, speeds up processing)
- **Storage**: 5GB free space
- **Camera**: HD webcam or smartphone

## Privacy & Data

### Data Storage
- **Local only**: All data stored on your computer
- **No cloud**: No data sent to external servers
- **User control**: You own your data

### Data Files
- `user.xlsx`: Usernames, passwords (encrypted recommended)
- `alpha_type.txt`: Current prediction result
- Videos: Not stored (processed and discarded)

### Security Recommendations
1. Use strong passwords
2. Don't share login credentials
3. Keep software updated
4. Backup `user.xlsx` regularly

## Support

### Getting Help
- **Technical Documentation**: See `TECHNICAL_DOCUMENTATION.md`
- **Issues**: Check troubleshooting section above
- **Questions**: Review this README thoroughly

### Reporting Bugs
When reporting issues, include:
1. Error message (exact text)
2. Steps to reproduce
3. Operating system
4. Python version
5. Screenshot (if applicable)

## Updates

### Version History
- **v1.0** (Current): Initial release
  - CV + NLP autism detection
  - 6 autism types
  - Personalized activities
  - Emotion detection

### Planned Features
- Multi-language support
- Mobile app version
- Cloud sync (optional)
- Progress reports
- Parent dashboard

## Credits

### Technologies Used
- **PyQt5**: User interface
- **PyTorch**: CV model framework
- **TensorFlow**: Emotion model framework
- **HuggingFace Transformers**: NLP model
- **OpenCV**: Video processing
- **BERT**: Language understanding
- **ResNet3D**: Video analysis

### Models
- **BERT-large**: Pre-trained by Google
- **ResNet3D**: Custom trained
- **Emotion Model**: Custom trained

## License

This project is for educational and research purposes.

## Disclaimer

**Important**: ALPHA is an AI-assisted tool and should NOT replace professional medical diagnosis. Always consult with qualified healthcare professionals for autism assessment and treatment.

The system provides:
- ✅ Preliminary screening
- ✅ Activity suggestions
- ✅ Progress tracking

The system does NOT provide:
- ❌ Medical diagnosis
- ❌ Treatment plans
- ❌ Professional medical advice

## Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] All packages installed (`pip install ...`)
- [ ] Model files in place
- [ ] `trained_models/` folder complete
- [ ] Run `python aplha_mian.py`
- [ ] Create account
- [ ] Upload video
- [ ] Answer questions
- [ ] View results
- [ ] Start activities

## Contact

For technical details, see: `TECHNICAL_DOCUMENTATION.md`

---

**ALPHA - Helping families understand and support autism spectrum disorders through AI technology.**

*Version 1.0 | 2024*
