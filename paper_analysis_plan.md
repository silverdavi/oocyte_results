# Oocyte Quality Assessment & IVF Counseling: A Dual-Model Approach for Transparent Patient Decision Support

## Paper Title: "Data-Driven IVF Counseling: Integrating Oocyte Quality Assessment with Personalized Cycle Predictions for Transparent Patient Decision Support"

## Core Message
**Transparent patient counseling through integrated data-driven models that provide realistic expectations at two critical decision points: before egg freezing (cycle planning) and after retrieval (quality assessment). The value is in revealing probabilities and attrition rates early enough to make informed decisions, not in improving biological outcomes themselves.**

## The Two-Model Framework

### Model 1: Oocyte Quality Assessment (Post-Retrieval Decision Support)
**Data Files:**
- `continuous_labels/blastulation_quality_scores.csv` - 704 embryo samples with continuous quality scores (0-1.0)
- `continuous_labels/continuous_cross_validation_results.csv` - 8-fold CV predictions vs actual outcomes
- `continuous_labels/continuous_training.log` - ViT-B-16 training details

**Architecture:** Vision Transformer (ViT-B-16) with frozen backbone + MLP classifier
**Purpose:** Real-time quality assessment of retrieved oocytes using microscopy images
**Clinical Value:** Replace vague "quality assessment" with quantitative predictions

### Model 2: Binary Classification Alternative
**Data Files:**
- `binary_labels/blastulation_binary_labels.csv` - Same 704 samples with binary outcomes
- `binary_labels/binary_cross_validation_results.csv` - Binary classification results
- `binary_labels/binary_training.log` + `binary_training_detailed.log` - Training logs

**Purpose:** Simpler binary classification for clinical environments preferring yes/no outcomes

### Model 3: Parametric Cycle Calculator (Pre-Cycle Planning)
**Data Files:**
- `calculator/parametric_cycle_calculator.py` - Sophisticated parametric model with age-based sigmoid functions

**Components:**
- AMH integration using age-specific percentile curves
- Age-dependent retrieval efficiency functions  
- Detailed attrition modeling through each IVF stage
- Live birth rate predictions with confidence intervals

**Purpose:** Transparent pre-cycle counseling replacing vague statistics with personalized predictions

## Research Priority Implementation Plan

### Priority 1: Model Performance Analysis ✅ READY
**Goal:** Demonstrate both models work well on real clinical data
- **Continuous model:** Correlation analysis, MAE, calibration plots
- **Binary model:** ROC curves, precision-recall, confusion matrices  
- **Calculator model:** Validation against known clinical outcomes
- **Cross-validation:** 8-fold results showing generalization

### Priority 2: Clinical Decision Support Framework
**Goal:** Show how models integrate into actual clinical workflows
- **Pre-cycle counseling:** AMH + age → expected outcomes with confidence intervals
- **Post-retrieval assessment:** Image → quality score → downstream predictions
- **Transparent attrition:** Show probability at each stage (retrieval → fertilization → blastulation → live birth)

### Priority 3: Comparison with Current Practice
**Goal:** Demonstrate improvement over current "standard of care"
- **Before:** Vague statistics ("success rates vary by age")
- **After:** Personalized predictions with confidence intervals
- **Value proposition:** Better patient decision-making through transparency

### Priority 4: Validation and Reliability
**Goal:** Show models are clinically reliable
- **Cross-validation performance**
- **Calibration analysis** (are 70% predictions actually right 70% of the time?)
- **Error analysis** and failure modes

## Key Clinical Value Propositions

1. **Pre-Cycle Transparency:** Replace generic age-based statistics with personalized AMH + age predictions
2. **Post-Retrieval Assessment:** Automated quality scoring instead of subjective manual assessment  
3. **Detailed Attrition Modeling:** Show probability at each stage rather than just "final success rates"
4. **Informed Decision Making:** Patients can make better choices about proceeding with cycles
5. **Expectation Management:** Realistic predictions reduce disappointment and improve satisfaction

## Paper Structure Outline

1. **Introduction:** Problem of poor patient counseling in IVF
2. **Methods:** Dual-model approach (vision + parametric)
3. **Results:** Model performance and clinical validation
4. **Discussion:** Integration into clinical workflows
5. **Conclusion:** Improved transparency leads to better patient outcomes

## Next Steps
1. Analyze model performance metrics (Priority 1)
2. Create clinical workflow diagrams 
3. Develop comparison with current practice
4. Write manuscript focusing on transparency value proposition

## Figures

### Figure 1: Dual-Model Framework Overview
**Panel A**: Clinical decision timeline showing two intervention points
**Panel B**: Model 1 - Oocyte quality assessment workflow (image → ML prediction → quality score)
**Panel C**: Model 2 - Calculator interface showing AMH input → personalized predictions
**Panel D**: Integration of both models in clinical workflow

### Figure 2: Model Performance and Validation
**Panel A**: Oocyte quality model cross-validation results (n=704, 8-fold CV)
**Panel B**: Prediction accuracy by quality bins (high/medium/low)
**Panel C**: Calculator model validation against population data
**Panel D**: Correlation between predicted and actual outcomes

### Figure 3: Transparent Counseling Impact
**Panel A**: Traditional vs. data-driven pre-cycle counseling (AMH-based predictions)
**Panel B**: Post-retrieval quality distribution examples (real patient cases)
**Panel C**: Attrition rate transparency (egg → embryo → live birth probabilities)
**Panel D**: Patient decision-making scenarios with quality data

### Figure 4: Clinical Implementation and Outcomes
**Panel A**: Workflow integration diagram
**Panel B**: Treatment decision changes based on quality assessment
**Panel C**: Prediction accuracy validation in clinical use
**Panel D**: Cost-benefit and scalability analysis

## Methods Focus

### Statistical Analysis
- **Model 1**: Cross-validation, ROC/AUC analysis, calibration plots
- **Model 2**: Parameter optimization, confidence interval estimation, sensitivity analysis
- **Integration**: Decision tree analysis for treatment recommendations

### Validation Strategy
- **Internal Validation**: Cross-validation and hold-out testing
- **External Validation**: Independent clinic data (if available)
- **Clinical Validation**: Prospective outcomes tracking

## Discussion Points

### Primary Value Proposition
1. **Transparency**: Replace vague counseling with data-driven predictions
2. **Early Decision Support**: Information available when choices can still be made
3. **Personalized Medicine**: Move beyond population averages to individual predictions

### Clinical Implementation
1. **Immediate Utility**: Models provide actionable information at critical decision points
2. **Workflow Integration**: Minimal disruption to existing clinical practice
3. **Scalable Technology**: Both models deployable across different clinic settings

### Future Directions
1. **Continuous Learning**: Models improve with more data
2. **Additional Factors**: Integration of other biomarkers and patient factors
3. **Decision Support Tools**: AI-assisted treatment planning

## Conclusion
A dual-model approach that provides transparent, data-driven counseling at two critical decision points in IVF treatment, moving from population-based generalizations to personalized predictions that enable informed decision-making.

---

## Analysis Priority:
1. Validate Model 1 performance (cross-validation results analysis)
2. Demonstrate Model 2 predictions (AMH-based scenarios)
3. Show integration benefits (combined decision support)
4. Clinical implementation framework

## 1. Paper Structure Overview

### Abstract
- Background: Gap between general IVF statistics and personalized patient counseling
- Methods: Dual-model approach - ViT oocyte quality assessment + parametric cycle prediction calculator
- Results: 71.2% accuracy in oocyte quality prediction with integrated personalized cycle forecasting
- Conclusion: Enables transparent, data-driven counseling at critical decision points

### Introduction
- Current limitations in IVF patient counseling (hand-waving statistics vs personalized data)
- Two critical decision points lacking data-driven support:
  1. Pre-cycle: Should I freeze eggs now? What can I expect?
  2. Post-retrieval: What's the quality of my oocytes? What are my realistic chances?
- Need for transparent attrition rate communication
- Importance of expectation alignment vs outcome improvement

### Methods
- **Model 1**: ViT-based oocyte quality assessment (702 embryo images)
- **Model 2**: Parametric calculator for cycle predictions (AMH, age, BMI, ethnicity factors)
- Integration methodology for dual-model patient counseling
- Validation approach for both prediction accuracy and clinical utility

### Results
- **Oocyte Quality Model Performance**
- **Cycle Prediction Calculator Validation** 
- **Integrated Counseling Framework Examples**
- **Attrition Rate Communication**

### Discussion
- Clinical utility for transparent patient counseling
- Impact on informed decision-making
- Limitations and appropriate use cases
- Future directions for data-driven reproductive counseling

## 2. The Dual-Model Framework

### 2.1 Model 1: Oocyte Quality Assessment (Post-Retrieval)
**Purpose**: Automated quality assessment to replace subjective "you have X oocytes" with "you have X oocytes with predicted quality scores"

**Data Source**: 
- `labels (1).csv`: Continuous quality scores (0-1) for 702 embryo images
- `results (4).csv`: Model predictions vs ground truth with cross-validation

**Key Metrics**:
- Binary classification: 70.8% accuracy, 97.9% sensitivity 
- Continuous prediction: 71.2% accuracy, 95.4% sensitivity
- High sensitivity crucial for not discarding viable oocytes

### 2.2 Model 2: Cycle Prediction Calculator (Pre-Freezing)
**Purpose**: Personalized cycle predictions based on individual factors instead of population averages

**Data Source**: `calculator/model.py` with parametric functions for:
- Age-based oocyte yield: `oocytes_by_age_new(age)`
- AMH adjustment: `gompertz(amh_ratio)` 
- BMI optimization: `bmi_factor(bmi)`
- Ethnicity factors: Population-based adjustments
- Live birth probability: `lbr_by_age(age)`

**Key Features**:
- Multi-cycle predictions (up to 3 rounds)
- Detailed attrition rate modeling
- Probabilistic birth outcomes: P = 1-(1-p)^n

## 3. Analysis Tasks

### 3.1 Oocyte Quality Model Validation
```python
# From: binary_labels/ and continuous_labels/
analysis_tasks = [
    "Cross-validation stability (8-fold CV)",
    "ROC analysis and threshold optimization", 
    "Sensitivity/specificity trade-offs for clinical use",
    "Error analysis: false positive vs false negative impact",
    "Comparison with random/shuffle baselines"
]
```

### 3.2 Calculator Model Validation  
```python
# From: calculator/model.py functions
validation_tasks = [
    "Age curve validation (20-45 years)",
    "AMH adjustment factor verification", 
    "BMI impact curve (15-45 BMI)",
    "Ethnicity factor validation",
    "Multi-cycle probability modeling",
    "Attrition rate accuracy vs published data"
]
```

### 3.3 Integrated Framework Examples
```python
patient_scenarios = [
    "28yo, normal AMH: Pre-cycle expectations vs post-retrieval reality",
    "35yo, low AMH: Multi-cycle planning with quality assessment",
    "40yo, high BMI: Realistic timeline and quality predictions",
    "32yo, first cycle: Detailed attrition rate explanation"
]
```

## 4. Required Plots (4 Main Figures)

### Figure 1: Oocyte Quality Model Performance
**Type:** Model validation and clinical utility
```python
subplots = [
    "A) ROC curves: Binary vs Continuous models",
    "B) Sensitivity/Specificity trade-offs for clinical thresholds", 
    "C) Cross-validation stability (8-fold CV)",
    "D) Error cost analysis: FP vs FN clinical impact"
]
```
**Message:** High sensitivity model suitable for clinical decision support

### Figure 2: Calculator Model Components
**Type:** Parametric function validation
```python
subplots = [
    "A) Age-based oocyte yield curves (20-45 years)",
    "B) AMH adjustment factor (Gompertz function)",
    "C) BMI optimization curve (polynomial fit)",
    "D) Multi-cycle probability modeling"
]
```
**Message:** Data-driven parametric model captures known biological relationships

### Figure 3: Integrated Patient Counseling Examples  
**Type:** Real patient scenario walkthroughs
```python
subplots = [
    "A) Pre-cycle prediction: Expected oocytes vs AMH/age",
    "B) Post-retrieval assessment: Quality scores + realistic probabilities",
    "C) Multi-cycle planning: Cumulative success probabilities", 
    "D) Attrition rate visualization: From oocytes to live birth"
]
```
**Message:** Transparent, personalized counseling at critical decision points

### Figure 4: Clinical Decision Support Framework
**Type:** Implementation and workflow integration
```python
subplots = [
    "A) Decision timeline: Pre-cycle vs post-retrieval interventions",
    "B) Expectation alignment: Personalized vs population statistics",
    "C) Transparency metrics: Confidence intervals and uncertainty",
    "D) Clinical workflow integration points"
]
```
**Message:** Practical framework for data-driven reproductive counseling

## 5. Key Technical Details

### 5.1 Data Processing Pipeline
```python
# Oocyte Quality Assessment
oocyte_data = {
    'images': 702,
    'labels': 'continuous_scores_0_to_1', 
    'validation': '8_fold_cross_validation',
    'models': ['ViT_binary', 'ViT_continuous']
}

# Calculator Parameters  
calculator_params = {
    'age_range': (20, 45),
    'amh_percentiles': 'age_adjusted_population_data',
    'bmi_polynomial': 'degree_4_optimization',
    'ethnicity_factors': {'asian': 0.82, 'black': 0.8, 'other': 0.85}
}
```

### 5.2 Integration Logic
```python
def integrated_counseling(patient_data, cycle_stage):
    if cycle_stage == 'pre_freezing':
        return calculator_predictions(patient_data)
    elif cycle_stage == 'post_retrieval':
        oocyte_scores = quality_assessment(patient_data['images'])
        updated_probs = update_calculator(patient_data, oocyte_scores)
        return combined_counseling(oocyte_scores, updated_probs)
```

## 6. Statistical Analysis Plan

### 6.1 Primary Endpoints
- **Oocyte Model**: Sensitivity ≥95% for clinical utility
- **Calculator Model**: Correlation with published attrition rates
- **Integration**: Improved prediction accuracy post-retrieval vs pre-cycle

### 6.2 Secondary Endpoints  
- Cross-validation stability across both models
- Confidence interval coverage for predictions
- Patient scenario validation against expert assessment

### 6.3 Clinical Validation
- Retrospective validation on held-out patient cohorts
- Comparison with current counseling standards
- Expert clinician assessment of prediction utility

## 7. Expected Results & Impact

### 7.1 Technical Achievements
- **Oocyte Quality**: 71.2% accuracy with 95.4% sensitivity
- **Calculator**: Validated parametric model with realistic predictions
- **Integration**: Seamless workflow for dual decision support

### 7.2 Clinical Impact
- **Pre-Cycle**: Replace general statistics with personalized AMH-based predictions
- **Post-Retrieval**: Provide objective quality assessment beyond simple counts
- **Overall**: Enable truly informed consent through transparent probability communication

### 7.3 Patient Benefits
- Realistic expectation setting before significant time/financial investment
- Objective quality information to support decision-making
- Detailed attrition rate understanding for informed consent
- Personalized counseling based on individual factors vs population averages

## 8. Key Messages for Paper

1. **Dual Decision Points**: Critical need for data support at both pre-cycle and post-retrieval stages
2. **Transparency Over Outcomes**: Focus on informed decision-making, not biological improvement claims  
3. **Personalized vs Population**: Individual factor-based predictions vs hand-waving statistics
4. **Attrition Rate Communication**: Detailed, visual explanation of realistic success probabilities
5. **Clinical Integration**: Practical framework that enhances rather than replaces clinical judgment
6. **Validated Approach**: Both data-driven ML and parametric modeling with proper validation

## 9. Implementation Roadmap

### Phase 1: Model Validation (Current)
- [ ] Complete oocyte quality model analysis
- [ ] Validate calculator parameters against literature  
- [ ] Generate patient scenario examples
- [ ] Create integration framework

### Phase 2: Clinical Testing  
- [ ] Retrospective validation on patient cohorts
- [ ] Expert clinician evaluation
- [ ] Patient feedback on counseling clarity
- [ ] Workflow integration testing

### Phase 3: Deployment
- [ ] Clinical decision support system
- [ ] Patient education materials
- [ ] Training protocols for counselors
- [ ] Outcome tracking for continuous improvement

---

**Next Immediate Steps:**
1. Analyze oocyte quality model performance from results files
2. Validate calculator functions against known biological relationships  
3. Create patient scenario examples showing before/after counseling
4. Generate all figures showing integrated approach
5. Draft methods and results sections focusing on dual-model framework 