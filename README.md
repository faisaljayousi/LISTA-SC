# Learned ISTA - Sparse Coding

TODO

## Installation

To run this project, you need to have Python 3 installed. You can create a virtual environment and install the required packages using the following commands:

```bash
# Create a virtual environment
conda create -n [venv_name] python=3.12

# Activate the virtual environment
conda activate [venv_name]

# Install dependencies
pip install -r requirements.txt
```

## Repository Structure

```bash
├── figures                    # Directory for figures and diagrams
│   ├── block_diagram.jpg
│   └── ista_algorithm.jpg
├── LICENSE                    # License information
├── main.ipynb                 # Jupyter notebook for interactive usage
├── README.md                  # Documentation for the repository
├── requirements.txt           # Required packages for the project
└── src                        # Source code for the project
    ├── architecture.py        # Definition of the LISTA model architecture
    ├── ista.py               # Implementation of ISTA
    ├── simulate_data.py      # Data simulation functions
    └── train.py              # Functions for training the LISTA model
```

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- Gregor, K. & LeCun, Y. (2010). Learning Fast Approximations of Sparse Coding. In J. Fürnkranz & T. Joachims (eds.), ICML (p./pp. 399-406), : Omnipress. [https://icml.cc/Conferences/2010/papers/449.pdf](https://icml.cc/Conferences/2010/papers/449.pdf).

## Contact

If you have any questions or feedback, feel free to reach out to me at [faisal.jayousi@proton.me](emailto:faisal.jayousi@proton.me).
