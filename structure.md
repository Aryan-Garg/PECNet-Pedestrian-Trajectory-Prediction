# Paper Structure (& Meta Content)
---

### Abstract
1. Human trajectory prediction systems in autonomous systems (1-2 sentences at max!)
2. Importance of developing robust and transparent systems (Our main highlight)
[Losing the sanctity of AI research to just benchmark datasets? -- too harsh?]
3. Write about all the testing and the conclusion (need to have a hypothesis first! And how to frame it???)

---

### Hypotheses: (?)
(Overfitting)
---

### Main Pointers & Notes (Ideas: Dump)
1. Noticed standardization, which in itself is an incorrect practice here. Why? Models is inherently non-linear and being a generative one, VAE in this case is also stochastic. 
    a. Surprising (and a major fault): During prediction and testing loops, the **final losses were divided by the same data scale value which was > 1. Precisely it was 1.86** (Insert picture of their github repo maybe?) 
    b. => 3 Encoders, 1 Decoder and 4 MLP-Predictor networks are linear which use a basic non-linear activation function(ReLU). 
    c. There is no mention or proof of this in their paper either.     
    (f(ax) != af(x))

2. Initial Attempts to **understand** and **improve** the model. Created synthetic data to run more tests to gain an overview of the system and parameter correlations with noise.   

3. Now divide the paper into 2 broad sections: 
    A. Understanding - (3-4 sentences: (Logic&Motivation behind that) before starting any sections below) 
        -> Ablation studies.
        -> Talk about the apparent negative correlation between FDE and ADE. (Expectation and actual behaviour) -- (try to get the metric from the exps) 
        -> Also, talk about other experiments. Check saved models and Logs_LossPlots dirs. (Also, find a way to insert pictures more aesthetically in overleaf)
        -> Trajectory Clustering and Classification (Use it from the colab notebook)
            i) Standard K-Means (++ Frobenius norm based experiment )
            ii) Clustering based on bounding boxes
            iii) Clustering based on number of points and 
            iv) (Subjective) Overall nature of each trajectory in the test dataset  (manual sifting through 2829 trajectories)
            (Show results/clusters in a table)

        ++ Talk about the custom non-linearity metric(Abruptness-Score) for each trajectory(show spectrum of test-data maybe?)
        
        -> Decoupling the system. Trajectory predictor & final destination predictor. (Expected results in terms of higher loss but the jump that ADE took was unexpected (> 10 times))

        -> Implemented **beta-annealing** to counter the vanishing gradient problem, which was encountered on faster learning rates.  
        
        NOTE: ADAM --> High sensitivity for learning param. Shape  & behavious of error surface(intuition: creek in the plateau).

    B. Improving - 
        -> Data Augmentation:
            i) RL: Real Trajectory + bots in the scene 
            ii) Physics based ones 
            iii) Interaction model
            iv) (Find this one)
        -> Hyperparam Tuning
        -> Possible Future Addition: Add a CVAE to do a multi-modal prediction? Not sure about the idea. Refer to Amit sir's picture of the new model.
        -> Adding Beta-annealing. (Need to understand this in detail!!!)

---

### Some more thoughts:

> Tried to understand the hypersensitivity of trajectory predictor as it kept blowing up the ADE on even 1\% augmented datasets. 

> Hints of over-fitting: We trained and tested the model on the same test dataset(2829) trajectories under standard conditions and found that the FDE and ADE were significantly higher. This result should be the theoretical limit as it reduces the problem to a simple compression and decompression problem. The loss witnessed here is due to the encoding and the stochastic layer. This will anyhow get added to newer examples. (HOW TO PROVE THIS?)  

> We used SIRENs instead of the conventional ReLU activated MLPs to encode information in higher derivatives of temporal data as well. And the results came out to be similar to above. This is proved in the SIREN paper and shows not only the hyper-sensitivity and over-fitting of PECNet but also proves the superiority of (Natural) Sinusoidal Activated Networks.   

> Can we introduce a (novel) criteria to quantify the deployability of an AI model for real world applications. Case in point PECNet. !dsfaklns!

---  

### ToDos:
1. Check papers related to this and the group
2. Write the novel stuff first in Understanding.
3. Come up with hypotheses & the !word!

---

0. Noticed standardization, which in itself is an incorrect practice here\ref{hyper}. What utterly surprised us was the fact that during prediction and testing loops, the final losses were divided by the same data scale value which was $> 1$. Precisely it was $1.86$ (Insert picture of their github repo maybe?) It effectively meant that the authors of PECNet treated their 3 Encoders, 1 Decoder and 4 MLP-Predictor networks as linear which used a ReLU activation function. There is no mention or proof of this in their paper either.   

1. Ignoring the grave problems in 0; initially -> Attempts to correctly understand and improve the model. Created synthetic data to run more tests to gain an overview of the system and parameter correlations with noise. 

2. Tried to understand the hypersensitivity of trajectory predictor as it kept blowing up the ADE on even 1\% augmented datasets. To test our dataset we performed two clustering analyses: i) Standard K-Means ii) Clustering based on bounding boxes, number of points and overall nature of each trajectory in the test dataset (manual sifting through 2829 trajectories) -> Show results/clusters in a table. Also created a custom non-linearity metric(Abruptness-Score) for each trajectory(show spectrum of test-data maybe?).

3. We decoupled the trajectory predictor from the destination predictor as well. Expected results in terms of higher loss but the jump that ADE took was unexpected (> 10 times). Even implemented $\beta$-annealing to counter the vanishing gradient problem on faster learning rates. 

4. Hints of over-fitting started to dawn upon us. We trained and tested the model on the same test dataset(2829) trajectories under standard conditions and found that the FDE and ADE were significantly higher. This result should be the theoretical limit as it reduces the problem to a simple compression and decompression problem. The loss witnessed here is due to the encoding and the stochastic layer. This will anyhow get added to newer examples. (HOW TO PROVE THIS?)\ref{ofInd}  

5. We used SIRENs instead of the conventional ReLU activated MLPs to encode information in higher derivatives of temporal data as well. And the results came out to be similar to (4) above. This is proved in the SIREN paper and shows not only the hyper-sensitivity and over-fitting of PECNet but also proves the superiority of (Natural) Sinusoidal Activated Networks.   

6. We introduce a (novel) criteria to quantify the deployability of an AI model for real world applications. Case in point PECNet.\ref{depFitness}  
