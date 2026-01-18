import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from retrieval_model import FOP  # Make sure this path is correct
import os


# === STEP 1: Load the face & voice embeddings from CSV ===
def read_matching_csvs(face_file, voice_file):
    print("Loading face and voice test embeddings...")
    face_df = pd.read_csv(face_file, header=None)
    voice_df = pd.read_csv(voice_file, header=None)

    face_ids = face_df.iloc[:, -1].values
    voice_ids = voice_df.iloc[:, -1].values

    face_feats = face_df.iloc[:, :-1].astype(np.float32).values
    voice_feats = voice_df.iloc[:, :-1].astype(np.float32).values

    face_tensor = torch.from_numpy(face_feats)
    voice_tensor = torch.from_numpy(voice_feats)

    return face_tensor, voice_tensor, face_ids, voice_ids


# === STEP 2: Load the model from checkpoint ===
def load_model(checkpoint_path, input_dim_face, input_dim_voice, FLAGS):
    model = FOP(FLAGS, input_dim_face, input_dim_voice)
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if FLAGS.cuda else 'cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if FLAGS.cuda:
        model.cuda()
    return model


# === STEP 3: Get aligned (projected) embeddings ===
def get_projected_embeddings(model, face_tensor, voice_tensor, use_cuda=True):
    model.eval()
    if use_cuda:
        face_tensor = face_tensor.cuda()
        voice_tensor = voice_tensor.cuda()

    with torch.no_grad():
        _, face_proj, voice_proj = model(face_tensor, voice_tensor)

    return face_proj.cpu().numpy(), voice_proj.cpu().numpy()


# === STEP 4: Evaluate matching accuracy ===
def evaluate_matching(face_proj, voice_proj, face_ids, voice_ids, amount_of_identities):
    correct = 0
    total = 0

    for i in range(0, len(voice_proj), amount_of_identities):
        voice = voice_proj[i]
        faces = face_proj[i:i + amount_of_identities]
        ids = face_ids[i:i + amount_of_identities]

        sims = cosine_similarity([voice], faces)[0]
        pred = np.argmax(sims)

        if ids[pred] == voice_ids[i]:
            correct += 1
        total += 1

    acc = correct / total
    print(f"🎯 Top-1 Matching Accuracy: {acc:.4f}\n")
    return acc


# === MAIN SCRIPT ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default="best_models/MSM_checkpoint_0016.pth.tar", help='Path to model checkpoint.')
    parser.add_argument('--identities', type=int, default=10, help='Amount of identities.')
    parser.add_argument('--evaluate_folder', action='store_true', help='Evaluate multiple checkpoints in a folder.')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA if available.')
    parser.add_argument('--dim_embed', type=int, default=900, help='Embedding dimension.')
    parser.add_argument('--fusion', type=str, default='linear', help='Fusion type.')
    parser.add_argument('--face_file_template', type=str, default='features_ECAPA_FaceNet/face_matching_{}.csv', help='Template path for face CSV.')
    parser.add_argument('--voice_file_template', type=str, default='features_ECAPA_FaceNet/voice_matching_{}.csv', help='Template path for voice CSV.')
    parser.add_argument('--checkpoint_dir', type=str, default="best_linear_model_alpha_1.00", help='Directory with multiple checkpoints.')

    FLAGS = parser.parse_args()

    class FLAGS_WRAPPED:
        cuda = FLAGS.cuda
        dim_embed = FLAGS.dim_embed
        fusion = FLAGS.fusion

    print("🔁 Starting evaluation...")

    face_file = FLAGS.face_file_template.format(FLAGS.identities)
    voice_file = FLAGS.voice_file_template.format(FLAGS.identities)

    face_tensor, voice_tensor, face_ids, voice_ids = read_matching_csvs(face_file, voice_file)

    if FLAGS.evaluate_folder:
        checkpoint_files = sorted([
            f for f in os.listdir(FLAGS.checkpoint_dir)
            if f.endswith(".pth.tar") and f.startswith("checkpoint_")
        ])

        for ckpt in checkpoint_files:
            checkpoint_path = os.path.join(FLAGS.checkpoint_dir, ckpt)
            try:
                model = load_model(checkpoint_path, input_dim_face=512, input_dim_voice=192, FLAGS=FLAGS_WRAPPED)
                face_proj, voice_proj = get_projected_embeddings(model, face_tensor, voice_tensor, use_cuda=FLAGS.cuda)
                print(f"🧪 Evaluating {ckpt}...")
                acc = evaluate_matching(face_proj, voice_proj, face_ids, voice_ids, FLAGS.identities)
            except Exception as e:
                print(f"❌ Failed to evaluate {ckpt}: {e}")
    else:
        model = load_model(FLAGS.checkpoint_path, input_dim_face=512, input_dim_voice=192, FLAGS=FLAGS_WRAPPED)
        face_proj, voice_proj = get_projected_embeddings(model, face_tensor, voice_tensor, use_cuda=FLAGS.cuda)
        evaluate_matching(face_proj, voice_proj, face_ids, voice_ids, FLAGS.identities)
