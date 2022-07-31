import flappy as flp
import itertools
import torch
from collections import deque
from PIL import Image
import json
import os
import torch.nn.functional as F
import torch
import argparse
from torch import nn, optim, distributions


def save_float_image(filepath: str, tensor: torch.Tensor):
    # print('tensor.size()', tensor.size(), tensor.dtype)
    tensor = (tensor * 256).byte()
    tensor = tensor.transpose(0, 1)
    print('tensor.size()', tensor.size(), tensor.dtype)
    im = Image.fromarray(tensor.numpy())
    im.save(filepath)


class Net(nn.Module):
    def __init__(self, input_channels: int, input_width: int, input_height: int):
        super().__init__()
        w = input_width
        h = input_height
        layers = []
        layers.append(nn.Conv2d(
            kernel_size=3, padding=1, in_channels=input_channels, out_channels=16))
        layers.append(nn.MaxPool2d(kernel_size=2))
        w //= 2
        h //= 2
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(
            kernel_size=3, padding=1, in_channels=16, out_channels=16))
        layers.append(nn.MaxPool2d(kernel_size=2))
        w //= 2
        h //= 2
        layers.append(nn.ReLU())
        self.conv_net = nn.Sequential(*layers)
        self.output = nn.Linear(w * h * 16, 2)

    def forward(self, x):
        x = self.conv_net(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.output(x)
        return x


def run(args):
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    screen, movementInfo = flp.main()

    image_width = flp.SCREENWIDTH // args.average_pool_size
    image_height = flp.SCREENHEIGHT // args.average_pool_size

    reward_baseline = 5
    reward_multiplier = 0.1

    f_logfile = open(args.logfile, 'w')

    net = Net(
        input_channels=args.previous_frames + 1,
        input_width=image_width,
        input_height=image_height)
    opt = optim.RMSprop(lr=args.lr, params=net.parameters())
    batch_reward_sum = 0
    batch_loss_sum = 0
    for episode in itertools.count():
        flappy = flp.Flappy(movementInfo)
        end_game = False
        screenbuf, end_game, res = flappy.step(False)
        screenbuf_t = torch.from_numpy(screenbuf)
        frames_l = deque()
        for _ in range(args.previous_frames):
            frames_l.append(torch.zeros(image_width, image_height))
        m_log_prob_sum = 0.0
        episode_entropy = 0
        for step_num in itertools.count():
            screenbuf_t = (screenbuf_t / 255).mean(dim=-1)
            screenbuf_t = screenbuf_t
            with torch.no_grad():
                screenbuf_t = F.avg_pool2d(
                    screenbuf_t.unsqueeze(0).unsqueeze(0),
                    args.average_pool_size
                ).squeeze(0).squeeze(0)
            # [36][64]
            frames_l.append(screenbuf_t)
            while len(frames_l) > args.previous_frames + 1:
                frames_l.popleft()
            frames_t = torch.stack(list(frames_l), dim=0).unsqueeze(0)
            # frames_t is [1][3][36][64]
            action_logits = net(frames_t)
            action_probs = F.softmax(action_logits, dim=-1)

            m = distributions.Categorical(action_probs)
            action = m.sample()
            action_bool = action == 1
            m_log_prob = m.log_prob(action)
            m_log_prob_sum = m_log_prob_sum + m_log_prob

            # step_entropy = - (action_probs * action_probs.log()).sum()
            # episode_entropy = episode_entropy + step_entropy

            screenbuf, end_game, res = flappy.step(action_bool)
            if end_game:
                reward = res['score']
                break
            screenbuf_t = torch.from_numpy(screenbuf)
        normalized_reward = (reward - reward_baseline) * reward_multiplier
        rl_loss = - m_log_prob_sum * normalized_reward
        # entropy_loss = - args.ent_reg * episode_entropy
        loss = rl_loss
        batch_loss_sum += loss.item()
        batch_reward_sum += reward
        loss.backward()
        if (episode + 1) % args.grad_accum_steps == 0:
            opt.step()
            opt.zero_grad()
            b = episode // args.grad_accum_steps
            batch_loss = batch_loss_sum / args.grad_accum_steps
            batch_reward = batch_reward_sum / args.grad_accum_steps
            batch_normalized_reward = (batch_reward - reward_baseline) * reward_multiplier
            print(
                f'b={b} loss={batch_loss:.3f} reward={batch_reward:.3f}'
                f' normalized_reward={batch_normalized_reward:.3f}')
            f_logfile.write(json.dumps({
                'batch': b,
                'loss': batch_loss,
                'reward': batch_reward,
                'normalized_reward': normalized_reward
            }) + '\n')
            f_logfile.flush()
            batch_loss_sum = 0
            batch_reward_sum = 0
        if episode % args.save_every == 0:
            save_filepath = args.model_path_templ.format(episode=episode)
            torch.save(net, save_filepath)
            print(f'saved {save_filepath}')

# log the batch reward
# log the batch loss
# and print these


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous-frames', type=int, default=2)
    parser.add_argument(
        '--average-pool-size', type=int, default=8,
        help='how much to shrink the input image')
    parser.add_argument(
        '--grad-accum-steps', type=int, default=16,
        help='how many episodes to accumulate gradients over before backprop')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--logfile', type=str, default='tmp/log.txt')
    # parser.add_argument(
    #     '--ent-reg', type=float, default=0.001,
    #     help='higher numbers => more exploration; lower numbers => more exploitation')
    parser.add_argument('--model-path-templ', type=str, default='tmp/model_{episode}.pt')
    parser.add_argument('--save-every', type=int, default=1000)
    args = parser.parse_args()
    run(args)
