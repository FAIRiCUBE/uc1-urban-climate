from src.measurer import Measurer
from types import ModuleType
from src.models.LSTM_ALL import LSTM_model
"""Example use of the bidirectional LSTM model for gap filling.
It also measures the time and memory used by the program and saves the results in a csv file."""

def main():
    # Path where the data are stored (the use of the disk in this path is measured).
    # Use '/' to measure the entire disk.
    data_path = "/"
    measurer = Measurer()
    tracker = measurer.start(data_path=data_path)
    # example -> shape = [5490, 2170]
    shape = []

    # Start
    LSTM_model(
        data_path="./data/test/UrbanAudit_ESTAT.csv",
        out_dir="./data/test/results",
        bidirectional=True,
        sequence_length=3,  # Define the sequence length (number of time steps to look back)
        city_name="Helsinki",  # e.g., 'Helsinki' (to run for a specific city
    )

    # it is very important to use program_path = __file__
    measurer.end(
        tracker=tracker,
        shape=shape,
        libraries=[
            v.__name__
            for k, v in globals().items()
            if type(v) is ModuleType and not k.startswith("__")
        ],
        data_path=data_path,
        program_path=__file__,
        variables=locals(),
        csv_file="./data/test/results/BLSTM_benchmarks.csv",
    )


if __name__ == "__main__":
    main()
