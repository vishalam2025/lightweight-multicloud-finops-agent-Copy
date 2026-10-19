\# Lightweight Multi-Cloud FinOps Agent



\### Predictive Waste Detection and Container Benchmarking



A lightweight, data-driven FinOps analytics project designed to analyze cloud workload utilization, identify potentially idle resources, investigate unusual usage patterns, evaluate CPU utilization forecasting, and explore container benchmarking.



\## Overview



Cloud infrastructure can incur unnecessary costs when resources are underutilized or workload patterns are not monitored effectively. This project explores a reproducible approach to workload analysis using historical virtual machine traces, hybrid detection techniques, forecasting evaluation, and an interactive dashboard.



The project uses a local Docker-based workflow to make data processing and visualization easier to reproduce without requiring continuously running cloud infrastructure.



\## Objectives



\- Analyze historical virtual machine utilization traces.

\- Identify potentially idle workloads for further review.

\- Investigate unusual resource utilization patterns.

\- Compare simple CPU utilization forecasting approaches.

\- Run the analysis pipeline and dashboard using Docker.

\- Explore workload datasets from multiple cloud environments.



\## Key Features



\- \*\*Workload preprocessing:\*\* Prepare and validate historical utilization data.

\- \*\*Hybrid detection workflow:\*\* Combine rule-based, statistical, and machine-learning techniques where implemented.

\- \*\*Waste identification:\*\* Group potentially idle workloads by review priority.

\- \*\*Forecast evaluation:\*\* Compare forecasting methods using Mean Absolute Error (MAE).

\- \*\*Interactive dashboard:\*\* Explore detection results and analytical outputs with Streamlit.

\- \*\*Containerized execution:\*\* Use Docker and Docker Compose for a reproducible local environment.

\- \*\*Dataset exploration:\*\* Investigate Bitbrains, Azure, and Alibaba workload traces, with the extent of validation varying by dataset.



\## System Architecture



```text

Historical Workload Datasets

&#x20;           |

&#x20;           v

&#x20;   Data Preprocessing

&#x20;           |

&#x20;           v

&#x20;Feature Extraction and Analysis

&#x20;           |

&#x20;           v

&#x20; Hybrid Detection Workflow

&#x20;(Rules, Statistics, ML Methods)

&#x20;           |

&#x20;           v

&#x20;Waste Review and Forecast Evaluation

&#x20;           |

&#x20;           v

&#x20;  Streamlit Dashboard and Reports

```



Container benchmarking is a supporting component used to investigate workload and execution performance.



\## Technology Stack



| Component             | Technologies                  |

| --------------------- | ----------------------------- |

| Programming language  | Python                        |

| Data analysis         | pandas, NumPy                 |

| Machine learning      | scikit-learn, where used      |

| Dashboard             | Streamlit                     |

| Containerization      | Docker                        |

| Service orchestration | Docker Compose                |

| Input data            | Historical VM workload traces |



\## Dataset and Evaluation



The recorded Bitbrains evaluation configuration includes:



\- Virtual machines examined: 50

\- Eligible virtual machines: 47

\- Historical observation window: Days 0–25

\- Forecast evaluation window: Days 26–28



\### Forecasting Results



| Forecasting method  |             Recorded MAE |

| ------------------- | -----------------------: |

| Last-value baseline | 0.3783 percentage points |

| Linear forecast     | 0.7083 percentage points |



The last-value baseline achieved a lower MAE than the linear forecast in this experiment. This result illustrates why forecasting approaches should be compared against simple baselines before selecting a more complex method.



These results describe the recorded experiment and should not be interpreted as a guarantee of performance on other datasets.



\## Project Structure



```text

CLOUD DA/

├── data/

│   ├── raw/

│   │   ├── bitbrains/

│   │   ├── azure/

│   │   └── alibaba/

│   └── processed/

├── benchmark\_containers.py

├── benchmark\_workload.py

├── check\_alibaba\_coverage.py

├── check\_sampling.py

├── dashboard.py

├── evaluate\_bitbrains.py

├── evaluate\_alibaba.py

├── evaluate\_azure.py

├── evaluate\_hybrid\_final.py

├── extract\_alibaba\_sample.py

├── extract\_azure\_sample.py

├── inspect\_alibaba.py

├── inspect\_azure.py

├── preprocess\_bitbrains.py

├── realdatasettest.py

├── Dockerfile

├── compose.yaml

├── requirements.txt

├── .gitignore

└── README.md

```



The listed structure is illustrative. Exact files and dataset availability may vary by project version.



\## Getting Started



\### Prerequisites



Install the following:



\- Git

\- Docker Desktop

\- Docker Compose (included with current Docker Desktop versions)



\### Clone the repository



```bash

git clone https://github.com/vishalam2025/lightweight-multicloud-finops-agent-copy.git

cd lightweight-multicloud-finops-agent-copy

```



\### Build the Docker image



```bash

docker build --load -t finops-da3 .

```



\### Run the analysis pipeline



```bash

docker compose run --rm pipeline

```



\### Start the dashboard



```bash

docker compose up -d dashboard

```



Open the local URL configured for the dashboard in `compose.yaml` or the terminal output, commonly:



`http://localhost:8501`



If your Docker configuration uses different service names, ports, or commands, follow the actual `Dockerfile` and `compose.yaml` in the repository.



\## Data and Reproducibility



Large raw workload datasets are excluded from the Git repository to keep cloning and pushing manageable.



Before running the pipeline, obtain the required datasets from their authorized sources and place them in the expected local data directories. Review the dataset licenses and usage terms before redistributing any data.



\## Limitations



\- Results depend on dataset quality, workload selection, and evaluation design.

\- A potentially idle classification is a recommendation for review, not proof that a resource can safely be shut down.

\- Forecasting results from one workload trace may not generalize to other environments.

\- Local dataset analysis does not itself demonstrate a deployed integration with AWS, Azure, or Alibaba Cloud.

\- Actual cloud cost savings require cost data, pricing models, and operational validation.



\## Future Enhancements



\- Validate detection methods across additional workload datasets.

\- Incorporate cloud pricing and cost-aware recommendations.

\- Improve forecasting with broader evaluation and additional baselines.

\- Add optional live cloud monitoring integrations.

\- Enhance resource recommendations with explainability and operational safeguards.



|   |   |

| - | - |

|   |   |

|   |   |

|   |   |



\## Acknowledgements



This project uses historical workload traces and open-source software for educational and analytical purposes. Refer to the relevant dataset documentation and software licenses before reuse or redistribution.



\## License



A license has not yet been specified. Add an appropriate license before permitting others to reuse or redistribute this project.



