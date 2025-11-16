# Dust and Scratch Removal in Scanned Photos

## Project Description
### Summary
This project restores old scanned photographs by removing dust, scratches, and cracks using image processing. It detects damaged regions through morphological operations and repairs them using multi-pass inpainting, producing a cleaner and visually improved image.

---

## Course concepts used
1. **Median Filtering**  
   Removes salt-and-pepper noise while preserving edges.

2. **Morphological Operations**  
   Top-hat, black-hat transforms, opening, dilation—used to highlight and isolate dust and scratch artifacts.

3. **Thresholding & Connected Component Analysis**  
   Used to generate and refine the damage mask by filtering small or irrelevant regions.

---

## Additional concepts used
1. **Multi-Pass Inpainting (Telea / Navier–Stokes)**  
   Iterative inpainting with multiple radii to fill long cracks and thin scratches effectively.

2. **Region-Based Geometric Filtering**  
   Uses blob area, thinness ratio, and axis lengths to selectively keep only true dust/scratch regions.

---

## Dataset
Scanned old photographs (user-provided).  
No external dataset required.

---

## Novelty
- **Adaptive Artifact Detection Pipeline:**  
  Combines top-hat & black-hat morphology with geometric filtering for precise detection of dust and scratches.

- **Shape-Aware Mask Refinement:**  
  Utilizes blob geometry metrics (area, thinness ratio, major/minor axis lengths) to retain only real scratches.

- **Multi-Pass Inpainting Strategy:**  
  Performs iterative inpainting with increasing radii for smoother restoration of long cracks.

- **Fully Automated Restoration:**  
  No manual marking required; suitable for batch restoration.

---

## Contributors
- Nilaya Reddy (PES1UG23EC206)
- Soujanya R H (PES1UG23EC303)
- Jammuladinne Harshithaa (PES1UG23EC130)
  
---

## Steps
### 1. Clone Repository
```bash
git clone https://github.com/DIP-2025-MINI-PROJECT/Dust-and-Scratch-Removal-in-Scanned-Photos
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Code
```bash
python restore_scratches_improved.py --input input/scanned.jpg --output output/cleaned.jpg --show
```

### Outputs

**Important intermediate steps**
- Damage mask generated using morphological operations
- Refined mask using geometric and region-based filtering
- Multi-pass inpainting results (optional)

**Final output images**
- Restored image: output/cleaned.jpg
- Mask used: output/cleaned_mask.png

---

## References
1. Gonzalez, R. C., & Woods, R. E. *Digital Image Processing*, Pearson  
2. OpenCV Documentation – Inpainting, Morphology, Filtering  
3. scikit-image Documentation – Morphology, Regionprops  
4. Bertalmio et al., *Image Inpainting*, SIGGRAPH (2000)

---

## Limitations and Future Work

### Limitations
- May miss extremely faint scratches  
- Cannot recover completely missing or torn image regions  

### Future Work
- Deep learning–based automated mask generation  
- Adaptive crack-width estimation  
- Color restoration for aged photographs  


