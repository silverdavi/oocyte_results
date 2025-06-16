# Data-Driven IVF Counseling: Oocyte Quality Assessment with Personalized Cycle Predictions

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](main.pdf)
[![Data](https://img.shields.io/badge/Data-702_samples-blue)](https://zenodo.org/records/6390798)
[![License](https://img.shields.io/badge/License-Academic-green)](LICENSE)

**Authors:** David Silver, Gilad Rave  
**Institution:** Rhea Labs ⊂ Rhea Fertility, Singapore  
**Paper:** 19 pages, 5 figures, 42 references  

---

## 📄 Paper Summary

This research presents a **dual-model framework** that revolutionizes IVF counseling by combining:

1. **🧮 Parametric Calculator**: Age-dependent AMH interpretation for transparent oocyte yield predictions
2. **🔬 AI Quality Assessment**: Vision Transformer model analyzing post-ICSI oocyte morphology for blastulation prediction

### Key Innovation
- **Post-ICSI, pre-2PN analysis**: Earliest possible oocyte quality prediction
- **Real clinical data**: 702 embryo samples from standardized time-lapse imaging
- **Evidence-based counseling**: Replacing population averages with personalized predictions

---

## 🎯 Key Results

| **Metric** | **Parametric Calculator** | **Oocyte Quality Model** |
|------------|---------------------------|---------------------------|
| **Purpose** | Oocyte yield prediction | Blastulation success prediction |
| **Input** | Age + AMH levels | Post-ICSI oocyte images |
| **Output** | Transparent yield estimates | Quality scores + classification |
| **Performance** | Age-dependent percentiles | r=0.421, 71.1% accuracy, 97.6% sensitivity |
| **Clinical Value** | Immediate counseling tool | Objective embryologist support |

---

## 📊 Publication Figures

### Figure 1: Parametric Calculator - AMH vs Oocyte Predictions
![AMH Calculator](figures/calculator_amh_oocytes.png)
*Age-dependent AMH interpretation showing personalized oocyte yield predictions across different AMH percentiles*

### Figure 2: Parametric Calculator - Age vs Oocyte Predictions  
![Age Calculator](figures/calculator_age_oocytes.png)
*Age-stratified oocyte yield analysis demonstrating declining fertility patterns with clinical thresholds*

### Figure 3: Oocyte Quality - Prediction Correlation
![Correlation Analysis](figures/oocyte_correlation.png)
*Correlation between predicted quality scores and actual blastulation outcomes (r=0.421, p<0.001)*

### Figure 4: Oocyte Quality - ROC Performance Comparison
![ROC Comparison](figures/oocyte_roc_comparison.png)
*Model performance vs random baseline with cross-validation analysis (AUC=0.661 vs 0.5)*

### Figure 5: Oocyte Quality - Classification Metrics
![Classification Metrics](figures/oocyte_classification_metrics.png)
*Comprehensive performance metrics: 71.1% accuracy, 97.6% sensitivity, 40.8% specificity*

---

## 📁 Complete Repository Structure

```
📦 oocyte_paper/
├── 📄 main.tex                           # Main LaTeX document
├── 📕 main.pdf                           # Compiled paper (19 pages)
├── 📂 sections/                          # Paper sections
│   ├── introduction.tex                  # Problem statement & motivation
│   ├── methods.tex                       # Methodology & implementation
│   ├── results.tex                       # Results & performance metrics
│   ├── discussion.tex                    # Clinical implications
│   └── conclusion.tex                    # Summary & future work
├── 🖼️ figures/                           # Publication-ready figures
│   ├── calculator_amh_oocytes.png        # AMH vs oocyte predictions
│   ├── calculator_age_oocytes.png        # Age vs oocyte predictions
│   ├── oocyte_correlation.png            # Quality correlation analysis
│   ├── oocyte_roc_comparison.png         # ROC performance comparison
│   └── oocyte_classification_metrics.png # Classification performance
├── 📚 references.bib                     # 42 comprehensive references
├── 📊 binary_labels/                     # Binary classification data & results
│   ├── blastulation_binary_labels.csv    # Binary outcome labels (702 samples)
│   ├── binary_cross_validation_results.csv # 8-fold CV results
│   ├── binary_training.log               # Training logs
│   └── *.py                              # Training & inference scripts
├── 📈 continuous_labels/                 # Continuous quality scores
│   ├── blastulation_quality_scores.csv   # Quality scores (0-1 range)
│   ├── continuous_cross_validation_results.csv # CV performance
│   └── continuous_training.log           # Training records
├── 🧮 calculator/                        # Parametric calculator
│   └── parametric_cycle_calculator.py    # Age-dependent AMH calculator
├── 🔬 data_analysis/                     # Analysis pipeline
│   ├── analyze_binary_model.py           # Binary model analysis
│   ├── analyze_continuous_model.py       # Continuous model analysis  
│   ├── analyze_parametric_calculator.py  # Calculator analysis
│   ├── run_all_analyses.py              # Complete pipeline runner
│   └── processed/                        # Processed results & metrics
├── 📊 plotting/                          # Figure generation scripts
│   ├── plot_calculator_focused.py        # Calculator visualization
│   ├── plot_oocyte_quality_focused.py   # Quality assessment plots
│   ├── plot_binary_model.py             # Binary model figures
│   ├── plot_continuous_model.py         # Continuous model figures
│   ├── create_all_plots.py              # Generate all figures
│   └── figures/                          # Generated figure outputs
├── 🐍 requirements.txt                   # Python dependencies
├── 📋 paper_analysis_plan.md            # Detailed methodology
├── 📝 references_validated.md           # Reference validation notes
└── 🎨 sn-*.bst                          # Springer Nature styles
```

---

## 🔬 Methodology

### Parametric Calculator
- **Age-dependent AMH percentiles** from clinical population data
- **Transparent mathematical relationships** avoiding black-box complexity
- **Sigmoid functions** modeling oocyte yield based on established literature

### Vision Transformer Model  
- **Architecture**: ViT-Base/16 with 86M parameters
- **Input**: 224×224 post-ICSI oocyte images (pre-2PN timing)
- **Training**: 8-fold cross-validation on 702 real clinical samples
- **Dataset**: [Gomez et al. 2022](https://zenodo.org/records/6390798) time-lapse embryo data

### Statistical Validation
- **Cross-validation**: 8-fold stratified sampling
- **Baselines**: Random, majority class, label-shuffled comparisons  
- **Significance testing**: Mann-Whitney U tests with Cohen's d effect sizes

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/silverdavi/oocyte_paper.git
cd oocyte_paper
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Complete Analysis
```bash
# Generate all processed results
cd data_analysis/
python run_all_analyses.py

# Create all figures  
cd ../plotting/
python create_all_plots.py
```

### 4. Compile Paper
```bash
# Generate final PDF
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 📈 Clinical Impact

### Current Problems Solved
- **❌ Population-based counseling** → **✅ Personalized predictions**
- **❌ Subjective embryo assessment** → **✅ Objective AI-assisted evaluation**  
- **❌ Fixed AMH thresholds** → **✅ Age-dependent interpretation**
- **❌ Late-stage decision making** → **✅ Early post-ICSI assessment**

### Implementation Benefits
- **🎯 Evidence-based counseling** with realistic expectations
- **⚡ Real-time predictions** for immediate clinical use
- **📊 Transparent methodology** for regulatory compliance
- **🔬 Objective quality assessment** reducing inter-observer variability

---

## 📊 Dataset Information

**Source**: [Gomez et al. 2022 Time-lapse Embryo Dataset](https://zenodo.org/records/6390798)
- **📹 704 time-lapse videos** with 2.4M images across 7 focal planes
- **🏥 Clinical data**: ICSI cycles from University Hospital of Nantes (2011-2019)
- **🔬 Acquisition**: Embryoscope© system with standardized protocols
- **🎯 Our subset**: 702 samples with complete blastulation outcomes

---

## 🔗 Related Publications

This research builds upon our previous work:
- **[Fordham et al. 2022](https://pubmed.ncbi.nlm.nih.gov/35944167/)** - *Human Reproduction*: Inter-observer variability in embryo assessment
- **[Rave et al. 2024](https://dl.acm.org/doi/10.1007/978-3-031-67285-9_12)** - *AIiH Conference*: Bonna Algorithm for implantation prediction

---

## 📋 Citation

```bibtex
@article{silver2024ivf,
    title={Data-Driven IVF Counseling: Integrating Oocyte Quality Assessment with Personalized Cycle Predictions},
    author={Silver, David H and Rave, Gilad},
    journal={In Preparation},
    year={2024},
    institution={Rhea Labs, Rhea Fertility},
    note={Repository: https://github.com/silverdavi/oocyte_paper}
}
```

---

## 📞 Contact

**🔬 Research Questions:**
- **David Silver**: david.silver@rhea-fertility.com
- **Gilad Rave**: gilad.rave@rhea-fertility.com

**🏢 Institution**: Rhea Labs ⊂ Rhea Fertility, Singapore

**🤝 Collaborations**: Open to academic and clinical partnerships

---

## 📜 License

This research is intended for **academic and clinical research purposes**. 
Please contact the authors for collaboration opportunities and commercial licensing.

---

<div align="center">

**Rhea Labs** ⊂ **Rhea Fertility**  
*Advancing reproductive medicine through data-driven innovation*

[🌐 Website](https://rhea-fertility.com) | [📧 Contact](mailto:david.silver@rhea-fertility.com) | [📄 Paper](main.pdf)

</div> 