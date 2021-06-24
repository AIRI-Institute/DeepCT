def get_embeddings(embeddings_path):
    """
    Fetches all embeddings used for training the model

    Returns
    -------
    embeddings
        Torch tensor.
    """
    rows = []
    with open(embeddings_path) as csvfile:
        csvreader = csv.reader(csvfile)

        for row in csvreader:
            rows.append(row)

    embeddings = []
    for row in rows:
        embeddings.append(list(map(float, row[1:])))

    embeddings = tf.convert_to_tensor(np.array(embeddings, dtype="float32"))
    embeddings = tf.expand_dims(embeddings, axis=1)
    return embeddings


def create_validation_set(sampler, batch_size=1, n_samples=None):
    """
    Generates the set of validation examples.

    Parameters
    ----------
    n_samples : int or None, optional
        Default is `None`. The size of the validation set. If `None`,
        will use all validation examples in the sampler.

    """
    (validation_batches, 
     validation_targets, 
     validation_cell_targets) = sampler.get_data_and_targets(batch_size, n_samples)
    
    return validation_batches


def save_checkpoint(manager, is_best=False):
    # Save the weights
    if is_best:
        manager.save()
    else:
        # currently unused
        checkpoint_name = "checkpoint"
        save_prefix = os.path.join(save_path, checkpoint_name)
        checkpoint.save(save_prefix)


def log_metrics(writer, step, train_loss, validation_loss, train_metrics, validation_metrics):
    writer.add_scalar("loss/train", train_loss, step)
    writer.add_scalar("loss/test", validation_loss, step)
    
    (t_r2_seq, 
     t_sp_seq, 
     t_pr_seq, 
     t_r2_cell, 
     t_sp_cell, 
     t_pr_cell) = train_metrics
    
    writer.add_scalar("across_sequence/r2/train", t_r2_seq, step)
    writer.add_scalar("across_sequence/spearman/train", t_sp_seq, step)
    writer.add_scalar("across_sequence/pearson/train", t_pr_seq, step)
    writer.add_scalar("across_cell_types/r2/train", t_r2_cell, step)
    writer.add_scalar("across_cell_types/spearman/train", t_sp_cell, step)
    writer.add_scalar("across_cell_types/pearson/train", t_pr_cell, step)
    
    (v_r2_seq,
     v_sp_seq,
     v_pr_seq,
     v_r2_cell,
     v_sp_cell,
     v_pr_cell) = validation_metrics
    
    writer.add_scalar("across_sequence/r2/test", v_r2_seq, step)
    writer.add_scalar("across_sequence/spearman/test", v_sp_seq, step)
    writer.add_scalar("across_sequence/pearson/test", v_pr_seq, step)
    writer.add_scalar("across_cell_types/r2/test", v_r2_cell, step)
    writer.add_scalar("across_cell_types/spearman/test", v_sp_cell, step)
    writer.add_scalar("across_cell_types/pearson/test", v_pr_cell, step)
    
    
def calc_metrics(targets, predictions):
    # inputs are matrices (n_sequences, n_cell_types)
    
    # across sequences (per cell type)
    r2_scores = []
    spearmanr_scores = []
    pearsonr_scores = []

    for i in range(targets.shape[1]):
        preds = predictions[:, i]
        tars = targets[:, i]
        
        if not np.all(tars==tars[0]):
            r2_scores.append(sklearn.metrics.r2_score(tars, preds))
            spearmanr_scores.append(scipy.stats.spearmanr(preds, tars)[0])
            pearsonr_scores.append(scipy.stats.pearsonr(preds, tars)[0])

    r2_seq_mean = np.mean(r2_scores)
    spearmanr_seq_mean = np.mean(spearmanr_scores)
    pearsonr_seq_mean = np.mean(pearsonr_scores)

    # across cell types (per sequence): 
    r2_scores = []
    spearmanr_scores = []
    pearsonr_scores = []

    for i in range(targets.shape[0]):
        preds = predictions[i, :]
        tars = targets[i, :]

        if not np.all(tars==tars[0]):
            r2_scores.append(sklearn.metrics.r2_score(tars, preds))
            spearmanr_scores.append(scipy.stats.spearmanr(preds, tars)[0])
            pearsonr_scores.append(scipy.stats.pearsonr(preds, tars)[0])

    r2_cell_mean = np.mean(r2_scores)
    spearmanr_cell_mean = np.mean(spearmanr_scores)
    pearsonr_cell_mean = np.mean(pearsonr_scores)
    
    return r2_seq_mean, spearmanr_seq_mean, pearsonr_seq_mean, r2_cell_mean, spearmanr_cell_mean, pearsonr_cell_mean