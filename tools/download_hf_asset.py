import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request


def sha256sum(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False,
                                     dir=os.path.dirname(output)) as tmp:
        tmp_path = tmp.name
    try:
        with urllib.request.urlopen(url) as response, open(tmp_path, 'wb') as f:
            shutil.copyfileobj(response, f)
        os.replace(tmp_path, output)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser(
        description='Download a file from Hugging Face and optionally verify SHA256.'
    )
    parser.add_argument('--url', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--sha256', default=None, type=str)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.force:
        if args.sha256:
            current_sha = sha256sum(args.output)
            if current_sha == args.sha256:
                print(f'Skipping existing verified file: {args.output}')
                return
            raise ValueError(
                f'Existing file has unexpected SHA256: {current_sha}. '
                'Use --force to overwrite it.')
        print(f'Skipping existing file: {args.output}')
        return

    download_file(args.url, args.output)
    if args.sha256:
        current_sha = sha256sum(args.output)
        if current_sha != args.sha256:
            raise ValueError(
                f'SHA256 mismatch for {args.output}: expected {args.sha256}, got {current_sha}'
            )
    print(f'Downloaded: {args.output}')


if __name__ == '__main__':
    main()
