# Arabic-to-English Neural Machine Translation (NMT)

## 📋 Project Overview

This project implements a **Neural Machine Translation system** that translates Arabic sentences to English using a **Transformer architecture**. It's a comprehensive pattern recognition project that demonstrates the complete pipeline for building and training a modern deep learning model.

The project is part of a **Pattern Recognition** course (Level 4, Semester 2) and demonstrates key concepts in:
- Natural Language Processing (NLP)
- Deep Learning with Transformers
- Sequence-to-Sequence modeling
- Text preprocessing and tokenization

---

## 🎯 Project Goals

1. **Understand Arabic-to-English translation** using neural networks
2. **Learn the Transformer architecture** (Attention mechanism, Multi-head attention)
3. **Apply pattern recognition techniques** to sequence data
4. **Preprocess and tokenize** multilingual text data
5. **Train and evaluate** a neural machine translation model
6. **Analyze model performance** and generate translations

---

## 📁 Project Structure

The notebook is organized into **3 major phases**:

### **PHASE 1: Data Loading & Exploration**
- Downloads Arabic-English sentence pairs from a GitHub dataset
- Analyzes dataset statistics (sentence lengths, vocabulary sizes)
- Visualizes the distribution of sentence lengths
- Sets a maximum sentence length cutoff to optimize training

**Key Outputs:**
- Dataset visualization: `length_distribution.png`
- Statistics on 100,000+ sentence pairs

---

### **PHASE 2: Text Preprocessing & Tokenization**
- **Cleans Arabic text**: Removes non-Arabic characters, normalizes spacing
- **Cleans English text**: Converts to lowercase, removes punctuation
- **Filters pairs**: Removes sentences exceeding max length
- **Builds vocabularies**: Creates word-to-index mappings for both languages
- **Tokenizes sentences**: Converts words to numerical indices
- **Pads sequences**: Makes all sentences uniform length for batching
- **Splits data**: Creates train (80%), validation (10%), and test (10%) sets

**Key Concepts:**
- Special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
- Vocabulary building with minimum frequency filtering
- Sequence padding for batch processing
- Train/validation/test stratification

**Saved Files:**
```
ar_train.npy, en_train.npy
ar_val.npy, en_val.npy
ar_test.npy, en_test.npy
arabic_vocab.pkl, english_vocab.pkl
idx_to_arabic.pkl, idx_to_english.pkl
config.pkl
```

---

### **PHASE 3: Building the Transformer Model**
- **Loads PyTorch and preprocessed data**
- **Implements core Transformer components:**
  - Positional Encoding (encodes word positions)
  - Multi-Head Attention (learns relationships between words)
  - Feed-Forward Networks (non-linear transformations)
  - Encoder (processes Arabic input)
  - Decoder (generates English output)
- **Defines the full Transformer model**
- **Trains the model** with backpropagation
- **Evaluates on validation and test sets**
- **Generates translations** for new Arabic text

**Architecture Highlights:**
- Encoder-Decoder architecture
- Multi-head attention mechanisms
- Layer normalization
- Dropout for regularization
- GPU acceleration (CUDA support)

---

## 🛠️ Technologies & Libraries

| Library | Purpose |
|---------|---------|
| **PyTorch** | Deep learning framework |
| **NumPy** | Numerical computations |
| **Matplotlib** | Data visualization |
| **Requests** | Download datasets from GitHub |
| **Pickle** | Serialize Python objects |
| **Regex (re)** | Text pattern matching and cleaning |

---

## 📊 Dataset

**Source:** [Arabic-English NMT Dataset](https://github.com/SamirMoustafa/nmt-with-attention-for-ar-to-en)

**Dataset Statistics:**
- **Total sentence pairs**: 100,000+
- **Language pair**: Arabic ↔ English
- **Average sentence length**: ~8-10 words
- **Maximum sentence length**: 30 words (after filtering)
- **Arabic vocabulary**: ~15,000 unique words
- **English vocabulary**: ~12,000 unique words

---

## 🚀 How to Use

### **1. Prerequisites**
Ensure you have Python 3.8+ installed with required libraries:

```bash
pip install torch numpy matplotlib requests
```

### **2. Run the Notebook**
Open the notebook in Jupyter or VS Code:

```bash
jupyter notebook final_version_of_pattern_project_v1_1.ipynb
```

Or in VS Code:
- Open the `.ipynb` file and run cells sequentially

### **3. Execution Order**

Run the notebook in three phases:

1. **Phase 1**: Execute all cells in "Data Loading & Exploration"
   - Downloads dataset (~50 MB)
   - Generates visualizations
   
2. **Phase 2**: Execute all cells in "Text Preprocessing & Tokenization"
   - Cleans and processes text
   - Saves preprocessed data (~100+ MB total)

3. **Phase 3**: Execute all cells in "Building the Transformer Model"
   - Loads PyTorch and preprocessed data
   - Trains the Transformer model
   - Generates translations

---

## 🔍 Key Concepts Explained

### **Tokenization**
Converting words into numerical indices so the model can process them:
```
"مرحبا" → [1, 2541, 2]  (SOS, word_id, EOS)
```

### **Padding**
Making all sequences the same length by adding PAD tokens:
```
[1, 45, 78, 2] → [1, 45, 78, 2, 0, 0, 0]  (MAX_LEN = 7)
```

### **Transformer Attention**
Mechanism that allows the model to focus on relevant parts of input:
- **Query, Key, Value** vectors are computed from input
- **Attention weights** determine which words to pay attention to
- **Multi-head attention** learns multiple patterns simultaneously

### **Encoder-Decoder Architecture**
- **Encoder**: Reads Arabic sentence and creates context vectors
- **Decoder**: Uses context to generate English translation word-by-word

---

## 📈 Model Training

The model is trained using:
- **Loss function**: Cross-Entropy Loss (measures prediction errors)
- **Optimizer**: Adam (adaptive learning rate optimization)
- **Batch size**: Configurable (typically 32-64)
- **Number of epochs**: Configurable (typically 10-20)
- **Learning rate**: Configurable (typically 0.0001-0.001)

**Training Process:**
1. Forward pass: Arabic text → Encoder → Context vectors → Decoder → English prediction
2. Compute loss between predicted and actual English
3. Backward pass: Compute gradients
4. Update model weights

---

## 📊 Evaluation Metrics

The model is evaluated using:
- **Loss curves**: Training vs. validation loss over epochs
- **Perplexity**: Measures how well the model predicts the next word
- **BLEU Score**: Measures similarity between predicted and actual translations (0-100, higher is better)
- **Sample translations**: Qualitative analysis of translation quality

---

## 💡 Usage Examples

### **Example 1: Basic Translation**
```python
# After training
arabic_text = "مرحبا، كيف حالك؟"
predicted_english = model.translate(arabic_text)
# Output: "hello, how are you?"
```

### **Example 2: Batch Processing**
```python
# Translate multiple sentences at once
arabic_sentences = [...list of Arabic text...]
translations = model.translate_batch(arabic_sentences)
```

---

## 🔧 Configuration & Hyperparameters

You can adjust these in the code:

```python
MAX_LEN = 30              # Maximum sentence length in words
MAX_SEQ_LEN = 32          # With <SOS> and <EOS> tokens
BATCH_SIZE = 32           # Number of samples per batch
LEARNING_RATE = 0.0001    # Optimizer learning rate
NUM_EPOCHS = 20           # Number of training iterations
EMBEDDING_DIM = 512       # Dimension of word embeddings
NUM_HEADS = 8             # Number of attention heads
NUM_LAYERS = 6            # Number of encoder/decoder layers
DROPOUT = 0.1             # Dropout probability
```

---

## 📁 Generated Files

After running the notebook, the following files are created:

| File | Purpose |
|------|---------|
| `ara_.txt` | Raw Arabic-English dataset |
| `length_distribution.png` | Histogram of sentence lengths |
| `ar_train.npy`, `en_train.npy` | Training data (tokenized & padded) |
| `ar_val.npy`, `en_val.npy` | Validation data |
| `ar_test.npy`, `en_test.npy` | Test data |
| `arabic_vocab.pkl` | Arabic word-to-index mapping |
| `english_vocab.pkl` | English word-to-index mapping |
| `idx_to_arabic.pkl` | Arabic index-to-word mapping |
| `idx_to_english.pkl` | English index-to-word mapping |
| `config.pkl` | Model configuration and constants |
| `model_checkpoint.pth` | Trained model weights (if saved) |

---

## 🐛 Troubleshooting

### **Issue: Dataset download fails**
- **Solution**: Check internet connection, or manually download from the GitHub URL

### **Issue: Out of memory (OOM) error**
- **Solution**: Reduce `BATCH_SIZE` or `MAX_SEQ_LEN`

### **Issue: CUDA GPU not detected**
- **Solution**: Verify PyTorch installation with `torch.cuda.is_available()`

### **Issue: Low translation quality**
- **Solution**: Increase `NUM_EPOCHS`, `NUM_LAYERS`, or try a larger vocabulary

---

## 📚 Learning Resources

### **Transformer Architecture**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original paper
- [Jay Alammar's Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

### **Neural Machine Translation**
- [Sequence to Sequence Learning](https://arxiv.org/abs/1409.3215)
- [Attention Mechanisms](https://arxiv.org/abs/1409.0473)

### **PyTorch Tutorials**
- [PyTorch Official Tutorial](https://pytorch.org/tutorials/)
- [Sequence Models](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

---

## ✅ Checklist

- [ ] Install Python 3.8+
- [ ] Install PyTorch, NumPy, Matplotlib, Requests
- [ ] Open notebook in Jupyter/VS Code
- [ ] Run Phase 1 (Data Loading & Exploration)
- [ ] Run Phase 2 (Preprocessing & Tokenization)
- [ ] Run Phase 3 (Build & Train Transformer)
- [ ] Evaluate model on test set
- [ ] Generate sample translations

---

## 📝 Notes

- **Training time**: Depends on hardware (GPU: ~30 min - 2 hours, CPU: several hours)
- **Data size**: ~150-200 MB of files generated during execution
- **GPU acceleration**: Highly recommended for faster training
- **Reproducibility**: Use `random.seed()` and `torch.manual_seed()` for consistent results



## 📄 License

This project uses the public Arabic-English dataset from GitHub and demonstrates educational concepts in NLP and deep learning.

---

