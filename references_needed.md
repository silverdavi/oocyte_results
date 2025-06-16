# Reference Requirements for IVF Counseling Paper

## Introduction Section

### 1. IVF statistics and current practice limitations
> "In vitro fertilization (IVF) represents a critical intervention for couples facing infertility, yet current counseling approaches rely heavily on population-based statistics that inadequately capture individual patient variability. Traditional IVF counseling typically provides aggregate success rates stratified by broad demographic categories, failing to account for the complex interplay between patient-specific factors such as age, ovarian reserve markers, and individual oocyte quality characteristics."

**Need references for:**
- IVF as critical intervention for infertility (general reproductive medicine statistics)
- Current counseling approaches using population-based statistics
- Limitations of aggregate success rates in counseling

### 2. AMH and cycle prediction limitations
> "For cycle predictions, clinicians often rely on simplified age-based estimates that ignore the substantial individual variation in ovarian reserve as measured by anti-Müllerian hormone (AMH) levels. These population averages can mislead patients about their specific prognosis, particularly given the dramatic age-dependent changes in AMH percentiles that are rarely considered in counseling protocols."

**Need references for:**
- Current reliance on age-based estimates
- AMH as ovarian reserve marker
- Age-dependent AMH percentile changes
- Problems with current counseling protocols

### 3. Morphological evaluation limitations
> "For oocyte quality assessment, current practice relies primarily on morphological evaluation by embryologists—a subjective process with significant inter-observer variability and limited predictive accuracy. While blastulation rates provide some indication of developmental competence, the assessment typically occurs after critical treatment decisions have already been made."

**Need references for:**
- Current morphological evaluation practices
- Inter-observer variability in embryology
- Limited predictive accuracy of morphological assessment
- Timing issues with current assessment methods

### 4. AI/ML advances in medical imaging
> "Recent advances in artificial intelligence and machine learning offer promising avenues for improving IVF counseling through more personalized, data-driven approaches. Vision Transformer (ViT) models have demonstrated remarkable capabilities in medical image analysis, while parametric modeling approaches can incorporate established clinical relationships in transparent, interpretable frameworks."

**Need references for:**
- AI/ML advances in medical imaging
- Vision Transformer capabilities in medical applications
- Success of parametric modeling in clinical settings

## Methods Section

### 5. Clinical IVF success rates
> "Each sample included unique identifiers, continuous quality scores (range: 0.004-0.999, mean: 0.592 ± 0.287), and binary blastulation labels (66.7\% successful outcomes, reflecting typical clinical IVF success rates)."

**Need references for:**
- Typical clinical IVF blastulation success rates (~66.7%)

### 6. Age-dependent AMH literature
> "The calculator implements age-specific AMH percentile distributions derived from published reproductive medicine literature."

**Need references for:**
- Published AMH percentile distributions by age
- Age-specific AMH normal values

### 7. Specific AMH values by age
> "where AMH percentiles demonstrate dramatic age-related decline (e.g., median AMH $\approx$ 1.8 ng/mL at age 25 vs. $\approx$ 0.18 ng/mL at age 42)."

**Need references for:**
- Specific AMH values: 1.8 ng/mL at age 25
- Specific AMH values: 0.18 ng/mL at age 42

### 8. Vision Transformer in medical imaging
> "We implemented a Vision Transformer (ViT) model for oocyte quality assessment, leveraging the architecture's demonstrated effectiveness in medical image analysis."

**Need references for:**
- Original Vision Transformer paper (Dosovitskiy et al.)
- ViT applications in medical image analysis

### 9. Statistical methods
> "Statistical significance testing used Mann-Whitney U tests comparing model performance against label-shuffled controls, with effect size calculation using Cohen's d."

**Need references for:**
- Mann-Whitney U test methodology
- Cohen's d effect size calculation

## Results Section

### 10. Clinical representativeness claim
> "The binary classification distribution showed 66.7\% positive blastulation outcomes (468 successful, 234 unsuccessful), reflecting typical clinical IVF success rates and confirming the dataset's clinical representativeness."

**Need references for:**
- Literature confirming 66.7% as typical clinical success rate

### 11. Clinical thresholds
> "Clinical thresholds for good (>15) and poor (<5) prognosis are indicated along with the AMA (Advanced Maternal Age) threshold at 35 years."

**Need references for:**
- >15 oocytes as good prognosis threshold
- <5 oocytes as poor prognosis threshold  
- 35 years as Advanced Maternal Age threshold

### 12. Expected clinical relationships
> "The model captures the expected relationship where higher AMH values consistently predict better retrieval outcomes" and "The model successfully captures both natural aging effects and differential impacts based on ovarian reserve"

**Need references for:**
- AMH-oocyte yield relationships
- Ovarian aging effects on IVF outcomes

## Discussion Section

### 13. Current IVF counseling practices
> "Current IVF counseling relies heavily on population-based statistics and subjective embryologist assessments."

**Need references for:**
- Current counseling practice patterns
- Reliance on subjective embryologist assessments

### 14. Inter-observer variability
> "The ViT model provides consistent, reproducible oocyte quality scores independent of inter-observer variability that plagues morphological evaluation."

**Need references for:**
- Studies documenting inter-observer variability in embryology

### 15. Conservative approach rationale
> "The corresponding modest specificity (23.1\%) suggests the model errs on the side of inclusion rather than exclusion—a conservative approach appropriate for reproductive medicine where false negatives carry higher clinical costs than false positives."

**Need references for:**
- Clinical cost-benefit analysis of false negatives vs false positives in IVF

### 16. Multi-center validation needs
> "Multi-center validation studies should confirm performance generalizability across diverse clinical settings and patient populations."

**Need references for:**
- Requirements for multi-center validation in medical AI
- FDA or regulatory guidance on clinical decision support tools

### 17. Regulatory considerations
> "Clinical decision support tools require appropriate regulatory oversight to ensure patient safety and efficacy claims."

**Need references for:**
- FDA guidance on clinical decision support software
- Regulatory frameworks for medical AI tools

### 18. Responsible AI implementation
> "This work demonstrates the potential for evidence-based, data-driven approaches to enhance reproductive medicine while maintaining realistic expectations about AI capabilities."

**Need references for:**
- Guidelines for responsible AI in healthcare
- Examples of successful clinical AI implementation

## Conclusion Section

### 19. Current clinical practice limitations
> "The integrated approach offers meaningful improvements over current subjective assessment methods while acknowledging the inherent complexity of predicting biological outcomes."

**Need references for:**
- Limitations of current subjective assessment methods
- Biological complexity in embryo development prediction

### 20. Clinical translation requirements
> "Future clinical translation requires multi-center validation studies, appropriate regulatory oversight, and comprehensive clinician training programs."

**Need references for:**
- Requirements for clinical translation of AI tools
- Clinician training needs for medical AI adoption

## Priority References Needed

### High Priority (Essential for credibility):
1. **Current IVF counseling practices and limitations** (Introduction)
2. **AMH age-dependent percentiles with specific values** (Methods/Results) 
3. **Inter-observer variability in embryology** (Introduction/Discussion)
4. **Typical IVF success rates (~67%)** (Methods/Results)
5. **Vision Transformer original paper and medical applications** (Introduction/Methods)

### Medium Priority (Important for context):
6. **Clinical thresholds (>15, <5 oocytes; AMA at 35)** (Results)
7. **AI/ML advances in medical imaging** (Introduction)
8. **Regulatory guidance for clinical decision support** (Discussion)
9. **Current morphological evaluation practices** (Introduction)
10. **AMH-oocyte yield relationships** (Results/Discussion)

### Lower Priority (Good to have):
11. **Responsible AI implementation guidelines** (Discussion/Conclusion)
12. **Multi-center validation requirements** (Discussion)
13. **Cost-benefit analysis of false negatives vs positives in IVF** (Discussion)
14. **Requirements for clinical AI translation** (Conclusion)
15. **Statistical methodology references** (Methods) 