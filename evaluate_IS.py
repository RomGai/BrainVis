from torch_fidelity import calculate_metrics

def compute_inception_score(image_folder):
    metrics = calculate_metrics(input1=image_folder, isc=True, cuda=True)
    return metrics

# Example usage
image_folder = "picture-gene"
m = compute_inception_score(image_folder)
print(m)
