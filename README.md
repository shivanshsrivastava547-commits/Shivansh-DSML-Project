# Shivansh-DSML-Project

Mini project for **Data Science and Machine Learning**.

**Student:** Shivansh Srivastava  
**Registration number:** RA2411056030011

## Project

Exploratory data analysis of the early 2019-nCoV outbreak, following the same Module 1 workflow as the course notebook `MiniProject_COVID19_India_EDA.ipynb` (load → inspect → clean → groupby/pivot → 12 charts → written summary).

| File | What it is |
| --- | --- |
| `2019_nC0v_20200121_20200126 - SUMMARY.csv` | Johns Hopkins CSSE aggregated situation reports, 21–26 January 2020 |
| `MiniProject_COVID19_nCoV_EDA.ipynb` | Executable EDA notebook (NumPy, Pandas, Matplotlib, Seaborn) |
| `MiniProject_COVID19_nCoV_EDA_Report.pdf` | Written report with name, registration number, tables and the 12 charts |
| `cleaned_ncov_summary.csv` | Cleaned table written by the notebook |
| `requirements.txt` | Python dependencies |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook MiniProject_COVID19_nCoV_EDA.ipynb
```

Run all cells top to bottom. The CSV must sit in the same folder as the notebook.
