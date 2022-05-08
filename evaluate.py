import torch
from lib.SentenceVAE import SentenceVAE
import argparse
import numpy as np


def main(arguments=None):
    # some initialization
    parser = get_parser()
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)

    # model loading
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    checkpoint['hyper_parameters']['batch_size'] = args.batch_size
    model = SentenceVAE(**checkpoint['hyper_parameters'])
    model.configure_ema()
    model.load_state_dict(checkpoint['state_dict'])  # load pre-trained model state
    model.ema_assign()  # assign EMA
    model.to(device)  # move to cuda if available
    model.freeze()  # we do not use model for re-training

    # evaluate likelihood
    if args.iw_samples > 0:
        iw_logpx_list = []
        total_words = 0
        for batch_idx, batch in enumerate(model.test_dataloader()):
            sequence, sequence_len = batch_to_sequence(batch, device)
            total_words += torch.sub(sequence_len, 1).sum().item()  # do not take into account <SOS>
            iw_logpx_batch, _ = estimate_likelihood_of_a_batch(model, sequence, sequence_len, args.iw_samples)
            iw_logpx_list.append(iw_logpx_batch)
        iw_logpx = -torch.cat(iw_logpx_list)
        ppl = torch.exp(iw_logpx.sum() / total_words)
        print("Total words =", total_words)
        print("NLL estimated with", args.iw_samples, "samples is ", iw_logpx.mean().item())
        print("PPL = ", ppl.cpu().detach().item())

    # evaluate likelihood
    # if args.iw_samples > 0:
    #     outputs = []
    #     for batch_idx, batch in enumerate(model.test_dataloader()):
    #         logpxz, kl_z = estimate_likelihood_of_a_batch(model, batch, args.iw_samples, device)
    #        outputs.append({'logpxz': logpxz, 'kl_z': kl_z})
    #         print("Processing batch #", batch_idx)
    #    logpxz = torch.cat([x['logpxz'] for x in outputs]).mean()
    #    kl_z = torch.cat([x['kl_z'] for x in outputs]).mean()
    #    print("NLL(", args.iw_samples, " samples) = ", -logpxz + kl_z)

    # generate random sentences
    for i in range(args.num_samples):
        print("Sentence #", i, ": ", model.sample(50))

    # interpolation
    if args.interpolation_steps > 0:
        # Option 1: from the training data set
        batch = next(iter(model.train_dataloader()))
        sequence, sequence_len = batch_to_sequence(batch, device)
        z1 = model.get_latent_representation(sequence[0:1]).unsqueeze(0)
        z2 = model.get_latent_representation(sequence[1:2]).unsqueeze(0)
        # Option 2: randomly sampled z values
        # z1 = model.prior_model.sample_once(torch.zeros(1, checkpoint['hyper_parameters']['z_dim']*2, device=device))
        # z2 = model.prior_model.sample_once(torch.zeros(1, checkpoint['hyper_parameters']['z_dim']*2, device=device))
        # number of interpolation steps
        k = args.interpolation_steps
        for i in range(k):
            z = (z1 * i + (k - 1 - i) * z2) * 1.0 / (k - 1)
            print("Step #", i, ": ", model.sample(50, z=z))

    # interpolation
    if args.num_neighbors > 0:
        batch = next(iter(model.train_dataloader()))
        sequence, sequence_len = batch_to_sequence(batch, device)
        print("Original sentence:", model.sequence_to_sentence(sequence[0]))
        print("sequence_len=", sequence_len[0])
        z_params, _ = model.encode(sequence[0].unsqueeze(0))
        z_params = z_params.unsqueeze(0)
        for i in range(args.num_neighbors):
            z = model.prior_model.sample_once(z_params, sampling_temperature=1.0)
            print("Neighbor #", i, ": ", model.sample(50, z=z))


def get_parser():
    parser = argparse.ArgumentParser()
    # generating sentences configurations
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to checkpoint.')
    parser.add_argument('--num_samples', type=int, default=0, help='Number of random samples to be generated.')
    parser.add_argument('--interpolation_steps', type=int, default=0, help='Number of interpolation steps.')
    parser.add_argument('--num_neighbors', type=int, default=0, help='Number of neighbors.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--iw_samples', type=int, default=0, help='Number of importance-weighted samples.')
    return parser


def batch_to_sequence(batch, device):
    sequence, sequence_len = batch.src
    sequence = sequence.to(device)
    sequence_len = sequence_len.to(device)
    return sequence, sequence_len


def estimate_likelihood_of_a_batch(model, sequence, sequence_len, iw_samples):
        """ Log-likelihood estimation with importance sampling. """
        logpxz = []
        kld = []
        for i in range(iw_samples):
            output = model(sequence, sequence_len)
            logpxz.append(output['logpxz'])
            kld.append(output['kl_z'])
        logpxz = torch.cat(logpxz)
        kld = torch.cat(kld)
        # estimate log-likelihood
        iw_elbo = (logpxz - kld).view(iw_samples, -1)
        iw_logpx = torch.logsumexp(iw_elbo, dim=0) - np.log(iw_samples)
        # estimate kl divergence
        iw_kld = kld.view(iw_samples, -1)
        iw_kld = torch.logsumexp(iw_kld, dim=0) - np.log(iw_samples)
        return iw_logpx, iw_kld


if __name__ == '__main__':
    main()