
import pandas as pd 
from typing import List, Union, Dict, Any, Tuple

def generate_output_dataframe(
    data, 
    columns: List[str] = [
        "uniprot_id",
        "node_id",
        "y",
        "prediction",
        "y_hat",
    ],
):
    df = pd.DataFrame(data, columns=columns)
    return df




def flatten_predictions(predictions: List[List[Tuple[Any]]]):
    flat_list = []
    for batch in predictions: 
        flat_list.extend(batch)
    return flat_list