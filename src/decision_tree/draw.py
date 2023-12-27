import matplotlib.pyplot as plt

# Data for plotting
thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
f1_scores = {
    'Efficientnet': [0.01434412746157823, 0.38458544180615023, 0.404981050351922, 0.37017658600392417, 0.3493889288281812, 
                     0.33565351894818246, 0.30969845150774244, 0.27835051546391754, 0.13627254509018036],
    'Inceptionv3': [0.015371240427657135, 0.3427682737169518, 0.385929648241206, 0.37209302325581395, 
                    0.35156819839533193, 0.3102893890675241, 0.2713915298184961, 0.2222222222222222, 
                    0.1786743515850144],
    'VGG16': [0.015962892164412355, 0.2562759222877101, 0.3256860098965362, 0.2996563573883161, 
              0.29185185185185186, 0.26454183266932274, 0.23699914748508097, 0.18761726078799248, 
              0.03879310344827586],
    'Resnet50': [0.015460217547346916, 0.36628511966701355, 0.4028307022318999, 0.39080459770114945, 
                 0.3649222065063649, 0.33903576982892686, 0.2958579881656805, 0.2327272727272727, 
                 0.18882466281310212]
}

# Create the plot
plt.figure(figsize=(10, 6))
for model, scores in f1_scores.items():
    plt.plot(thresholds, scores, marker='o', label=model)

# Adding title and labels
plt.title('F1 Scores of Xgb Models Using Different Feature Extractors Across Thresholds')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.grid(True)
plt.legend()
plt.savefig('f1_scores.pdf')
plt.show()