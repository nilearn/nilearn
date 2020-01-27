import json
import os

from nilearn.datasets.utils import _get_dataset_dir


def make_fresh_openneuro_dataset_urls_index(
        data_dir=None,
        dataset_version='ds000030_R1.0.4',
        verbose=1,
        ):
    """ ONLY intended for Nilearn developers, not general users.
    Creates a fresh, updated OpenNeuro BIDS dataset index from AWS,
    ready for upload to osf.io .

    Crawls the server where OpenNeuro dataset is stored
    and makes a JSON file `nistats_fetcher_openneuro_dataset_urls.json'
    containing a fresh list of dataset file URLs.

    Note: Needs Python package `Boto3`.

    Do NOT rename this file.

    This file can now be uploaded to Quick-Files section
    of the Nilearn account on osf.io .

    Then this file can be downloaded by
    :func:`datasets.fetch_openneuro_dataset_index`

    Run this function and upload the new file if the URL index downloaded by
    :func:`datasets.fetch_openneuro_dataset_index` becomes outdated.

    This approach is faster than crawling the servers anew every time
    the OpenNeuro dataset is downloaded,
    and circumvents `boto3` as a dependency for everyday use.

    Parameters
    ----------
    data_dir: string, optional
        Path to store the downloaded dataset.
        If None downloads to user's Desktop

    dataset_version: string, optional
        dataset version name. Assumes it is of the form [name]_[version].

    verbose: int, optional
        verbosity level (0 means no message).

    Returns
    -------
    urls_path: string
        Path to downloaded dataset index

    urls: list of string
        Sorted list of dataset directories
    """
    import boto3
    from botocore.handlers import disable_signing
    if not data_dir:
        data_dir = os.path.expanduser('~/Desktop')
    data_prefix = '{}/{}/uncompressed'.format(
        dataset_version.split('_')[0], dataset_version)

    data_dir = _get_dataset_dir(data_prefix, data_dir=data_dir,
                                verbose=verbose)

    # First we download the url list from the uncompressed dataset version
    urls_path = os.path.join(data_dir,
                             'nistats_fetcher_openneuro_dataset_urls.json',
                             )
    urls = []
    if os.path.exists(urls_path):
        with open(urls_path, 'r') as json_file:
            urls = json.load(json_file)
        existing_index_msg = ("There is an existing url index at `{}`. "
                              "Aborting download of fresh index."
                              .format(urls_path)
                              )
        print(existing_index_msg)
    else:
        resource = boto3.resource('s3')
        resource.meta.client.meta.events.register('choose-signer.s3.*',
                                                  disable_signing)
        bucket = resource.Bucket('openneuro')

        for obj in bucket.objects.filter(Prefix=data_prefix):
            # get url of files (keys of directories end with '/')
            if obj.key[-1] != '/':
                url = '{}/{}/{}'.format(bucket.meta.client.meta.endpoint_url,
                                        bucket.name,
                                        obj.key,
                                        )
                urls.append(url)
        urls = sorted(urls)

        with open(urls_path, 'w') as json_file:
            json.dump(urls, json_file)
        print("Saved updated url index to {}.\nUpload it with the same name "
              "to the quick-files section of osf.io using the Nilearn account "
              "to update the file without breaking the fetcher download link."
              .format(urls_path))
    return urls_path, urls
