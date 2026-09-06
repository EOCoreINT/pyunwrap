import sys, time, torch, os, glob, json, argparse
sys.path.insert(0, '/home/claude/pyunwrap')
from pyunwrap.data.dataloader import InSARTileDataset
from pyunwrap.models.ambiguity_net import AmbiguityNet
from pyunwrap.training.trainer import (
    Trainer, evaluate, build_warmup_cosine_scheduler, build_curriculum_aware_scheduler,
)

parser = argparse.ArgumentParser()
parser.add_argument('run_name')
parser.add_argument('target_epoch', type=int)
parser.add_argument('--total-epochs', type=int, default=60)
parser.add_argument('--scheduler', choices=['buggy', 'fixed'], default='fixed')
parser.add_argument('--checkpoint-every', type=int, default=5)
args = parser.parse_args()

CKPT_DIR = f'{args.run_name}/checkpoints'
LOG_PATH = f'{args.run_name}/log.jsonl'
os.makedirs(CKPT_DIR, exist_ok=True)


def latest_checkpoint():
    ckpts = sorted(glob.glob(f'{CKPT_DIR}/epoch_*.pt'))
    return ckpts[-1] if ckpts else None


def save_checkpoint(trainer, epoch):
    tmp_path = f'{CKPT_DIR}/epoch_{epoch:04d}.pt.tmp'
    final_path = f'{CKPT_DIR}/epoch_{epoch:04d}.pt'
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'epoch': epoch,
    }, tmp_path)
    os.rename(tmp_path, final_path)
    for old in sorted(glob.glob(f'{CKPT_DIR}/epoch_*.pt'))[:-1]:
        os.remove(old)


def build_trainer():
    torch.manual_seed(42)
    train_ds = InSARTileDataset('train_nb.h5', augment=True, require_ground_truth=True, seed=42)
    val_ds = InSARTileDataset('val_nb.h5', augment=False, require_ground_truth=True, seed=43)
    model = AmbiguityNet(pretrained=False, k_max=10.0, dropout_rate=0.1)
    trainer = Trainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds,
        out_dir=args.run_name, total_epochs=args.total_epochs, warmup_epochs=5,
        base_lr=1e-4, batch_size=8, num_workers=0, grad_clip_norm=1.0,
        validate_every=args.total_epochs,  # only validate at the very end; we track train loss for the curve
        device='cpu', use_curriculum=True, stratified_validation=False,
    )
    if args.scheduler == 'buggy':
        # Force the OLD single-cycle scheduler, bypassing Trainer's default
        # curriculum-aware one -- this reproduces the exact bug being
        # documented, using the real (still-present, still-supported)
        # standalone scheduler function rather than a fabricated stand-in.
        trainer.scheduler = build_warmup_cosine_scheduler(trainer.optimizer, args.total_epochs, warmup_epochs=5)
    return trainer


if __name__ == '__main__':
    trainer = build_trainer()
    ckpt_path = latest_checkpoint()
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        trainer.model.load_state_dict(ckpt['model_state_dict'])
        trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from {ckpt_path} (epoch {ckpt["epoch"]}). Continuing from epoch {start_epoch}.')
    else:
        start_epoch = 1
        print(f'No checkpoint found for {args.run_name}. Starting fresh from epoch 1.')

    t0 = time.time()
    log_f = open(LOG_PATH, 'a')
    for epoch in range(start_epoch, args.target_epoch + 1):
        loader = trainer._build_epoch_loader(epoch)
        train_metrics = trainer._train_one_epoch(epoch, loader)
        trainer.scheduler.step()
        lr = trainer.optimizer.param_groups[0]['lr']
        loss = train_metrics['loss/total']
        print(f'[{args.run_name}] Epoch {epoch}/{args.total_epochs} loss={loss:.4f} lr={lr:.2e}')
        log_f.write(json.dumps({'epoch': epoch, 'loss': loss, 'lr': lr}) + '\n')
        log_f.flush()
        if epoch % args.checkpoint_every == 0 or epoch == args.target_epoch:
            save_checkpoint(trainer, epoch)
    log_f.close()
    print(f'Chunk to epoch {args.target_epoch} done in {time.time()-t0:.1f}s')
