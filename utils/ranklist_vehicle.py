import torch
import os
from utils.reranking import re_ranking
import numpy as np
from sklearn import manifold
import matplotlib.patches as patches
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.metrics.pairwise import cosine_similarity
from scipy.integrate import simps

def eval_func_msrv(distmat, q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)

    query_arg = np.argsort(q_pids, axis=0)
    result = g_pids[indices]
    gall_re = result[query_arg]
    gall_re = gall_re.astype(np.str_)
    # pdb.set_trace()

    result = gall_re[:, :100]

    # with open("re.txt", 'w') as file_obj:
    #     for li in result:
    #         for j in range(len(li)):
    #             if j == len(li) - 1:
    #                 file_obj.write(li[j] + "\n")
    #             else:
    #                 file_obj.write(li[j] + " ")
    with open('re.txt', 'w') as f:
        f.write('rank list file\n')

    # pdb.set_trace()
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        q_sceneid = q_sceneids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        # original protocol in RGBNT100 or RGBN300
        # remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)

        # for each query sample, its gallery samples from same scene with same or neighbour view are discarded # added by zxp
        # symmetrical_cam = (8 - q_camid) % 8
        # remove = (g_pids[order] == q_pid) & ( # same id
        #              (g_sceneids[order] == q_sceneid) & # same scene
        #              ((g_camids[order] == q_camid) | (g_camids[order] == (q_camid + 1)%8) | (g_camids[order] == (q_camid - 1)%8) | # neighbour cam with q_cam
        #              (g_camids[order] == symmetrical_cam) | (g_camids[order] == (symmetrical_cam + 1)%8) | (g_camids[order] == (symmetrical_cam - 1)%8)) # nerighboour cam with symmetrical cam
        #          )
        # new protocol in MSVR310
        remove = (g_pids[order] == q_pid) & (g_sceneids[order] == q_sceneid)
        keep = np.invert(remove)

        with open('re.txt', 'a') as f:
            f.write('{}_s{}_v{}:\n'.format(q_pid, q_sceneid, q_camid))
            v_ids = g_pids[order][keep][:max_rank]
            v_cams = g_camids[order][keep][:max_rank]
            v_scenes = g_sceneids[order][keep][:max_rank]
            for vid, vcam, vscene in zip(v_ids, v_cams, v_scenes):
                f.write('{}_s{}_v{}  '.format(vid, vscene, vcam))
            f.write('\n')

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        # tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

class R1_mAP():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.reset()

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        # Store image paths as simple names
        self.img_paths = []
        self.sceneids = []
        self.img_prefixes = {
            'vis': '/media/zpp2/Datamy/lyy/512/data/MSVR310/bounding_box_test/',
            'ni': '/media/zpp2/Datamy/lyy/512/data/MSVR310/bounding_box_test/',
            'th': '/media/zpp2/Datamy/lyy/512/data/MSVR310/bounding_box_test/'
        }

    def update(self, output):
        feat, pid, camid, sceneid, img_path = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.sceneids.extend(np.asarray(sceneid))
        self.img_paths.extend(img_path)

    def set_image_prefixes(self, rgb_prefix, nir_prefix, tir_prefix):
        self.img_prefixes['vis'] = rgb_prefix
        self.img_prefixes['ni'] = nir_prefix
        self.img_prefixes['th'] = tir_prefix

    def load_image(self, path):
        if not os.path.isfile(path):
            print(f"Warning: File {path} does not exist.")
            return np.zeros((100, 100, 3), dtype=np.uint8)  # Return a dummy image with default size
        image = Image.open(path).convert('RGB')  # Ensure image is in RGB format
        image = image.resize((256, 128))
        return np.array(image)  # Convert to NumPy array

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        '''
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])

        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        '''

        feats_query = feats
        feats_gallery = feats
        if self.feat_norm:
            print("The test feature is normalized")
            feats_query = torch.nn.functional.normalize(feats_query, dim=1, p=2)  # along channel
            feats_gallery = torch.nn.functional.normalize(feats_gallery, dim=1, p=2)

        # query
        qf = feats_query[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_sceneids = np.asarray(self.sceneids[:self.num_query])  # zxp

        # gallery
        gf = feats_gallery[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_sceneids = np.asarray(self.sceneids[self.num_query:])  # zxp

        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func_msrv(distmat, q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids)
        # Visualize top-20 results for each query
        self.visualize_ranked_list_results(distmat, topk=5, save_dir='zrankList/310/baseline+SIM+CLS+PAT')
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf
    
    def visualize_ranked_list_results(self, distmat, topk=5, save_dir='vis_results'):
        """
        Visualize the top-N ranked results for each query in RGB, NIR, and TIR modalities.
        :param distmat: Distance matrix between query and gallery.
        :param topk: Number of top ranked results to visualize.
        :param save_dir: Directory to save the visualized results.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(0, 200):
            query_name = self.img_paths[i]  # Simple image name (without prefix)
            query_pid = self.pids[i]  # Query person ID
            query_sceneid = self.sceneids[i]  # Query scene ID

            # Get top-20 ranked indices, considering only different cameras
            ranked_indices = [idx for idx in np.argsort(distmat[i]) if self.sceneids[idx + self.num_query] != query_sceneid]
            ranked_indices = ranked_indices[:topk]  # Get top-20 indices

            # Load query images
            ID = query_name.split('_')[0]
            query_imgs = [self.load_image(os.path.join(self.img_prefixes[modality]+f"{ID}/{modality}/", query_name))
                          for modality in ['vis', 'ni', 'th']]

            # Load gallery images
            gallery_paths = [self.img_paths[idx + self.num_query] for idx in ranked_indices]  # List of names
            gallery_img_ID = [path.split('_')[0] for path in gallery_paths]
            gallery_paths = [[os.path.join(self.img_prefixes[modality]+f"{gallery_img_ID[index]}/{modality}/", path) for index, path in enumerate(gallery_paths)]
                             for modality in  ['vis', 'ni', 'th']]

            # Prepare new gallery paths
            new_gallery_paths = []
            for j in range(topk):
                gallery_path = [gallery_paths[0][j], gallery_paths[1][j], gallery_paths[2][j]]
                new_gallery_paths.append(gallery_path)
            gallery_paths = new_gallery_paths

            gallery_pids = [self.pids[idx + self.num_query] for idx in ranked_indices]  # Get corresponding PIDs

            # Load gallery images
            gallery_imgs = [[self.load_image(path) for path in paths] for paths in zip(*gallery_paths)]

            # Plot and save
            self.plot_images(query_imgs, gallery_imgs, gallery_pids, query_pid,
                             save_path=os.path.join(save_dir, f'query_{i}_results.png'))

    def plot_images(self, query_imgs, gallery_imgs, gallery_pids, query_pid, save_path=None):
        """
        Helper function to plot the query images and the gallery images.
        Correctly retrieved images are marked with a green box, incorrect ones with a red box.
        """
        num_results = len(gallery_imgs[0])  # Number of top-ranked results
        fig, axs = plt.subplots(3, num_results + 1, figsize=(20, 6),
                                gridspec_kw={'wspace': 0.2, 'hspace': 0.1})

        # Display the query images
        modalities = ['RGB', 'NIR', 'TIR']
        for j, (img, modality) in enumerate(zip(query_imgs, modalities)):
            axs[j, 0].imshow(img)
            axs[j, 0].set_title(f"Query {modality}", fontsize=21)
            axs[j, 0].axis('off')

        # Display the top-ranked results
        for i, (imgs, pid) in enumerate(zip(zip(*gallery_imgs), gallery_pids)):  # Unzip and process each modality
            for j, (img, modality) in enumerate(zip(imgs, modalities)):
                axs[j, i + 1].imshow(img)
                axs[j, i + 1].axis('off')

                # Add a green or red rectangle around the image
                color = 'green' if pid == query_pid else 'red'
                rect = patches.Rectangle((0, 0), img.shape[1], img.shape[0], linewidth=10, edgecolor=color,
                                         facecolor='none')
                axs[j, i + 1].add_patch(rect)
                axs[j, i + 1].set_title(f"Rank {i + 1}", fontsize=22)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        # plt.show()
        plt.close()
    



class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.reset()

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        # Store image paths as simple names
        self.img_paths = []
        self.img_prefixes = {
            'RGB': './data/RGBNT201/test/RGB/',
            'NIR': './data/RGBNT201/test/NI/',
            'TIR': './data/RGBNT201/test/TI/'
        }

    def update(self, output):  # called once for each batch
        feat, pid, camid, img_paths = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        # img_paths should be a list of image names, not full paths
        self.img_paths.extend(img_paths)

    def set_image_prefixes(self, rgb_prefix, nir_prefix, tir_prefix):
        self.img_prefixes['RGB'] = rgb_prefix
        self.img_prefixes['NIR'] = nir_prefix
        self.img_prefixes['TIR'] = tir_prefix

    def load_image(self, path):
        if not os.path.isfile(path):
            print(f"Warning: File {path} does not exist.")
            return np.zeros((100, 100, 3), dtype=np.uint8)  # Return a dummy image with default size
        image = Image.open(path).convert('RGB')  # Ensure image is in RGB format
        return np.array(image)  # Convert to NumPy array

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])

        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        if self.reranking:
            print('=> Enter reranking')
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)

        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        # T-sne Visualization
        # self.showPointMultiModal(feats, real_label=self.pids,
        #                          draw_label=[258, 260, 269, 271, 273, 280, 282, 284, 285, 286, 287, 289],
        #                          save_path='zTSNE/201/baseline+SIM+CLS+PAT')
        
        # Visualize top10 results for each query
        #self.visualize_ranked_list_results(distmat, topk=10, save_dir='zrankList/201/baseline+SIM+CLS+PAT',num2vis=200)
        
        
        #self.plot_similarity_distribution(qf, q_pids,base_dir = 'zSimilarity_pic/201/',condition='baseline')
        

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf

    # 2 Rank_list
    def visualize_ranked_list_results(self, distmat, topk=10, save_dir='vis_results',num2vis=10):
        print("enter rank_list")
        """
        Visualize the top-N ranked results for each query in RGB, NIR, and TIR modalities.
        :param distmat: Distance matrix between query and gallery.
        :param topk: Number of top ranked results to visualize.
        :param save_dir: Directory to save the visualized results.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        num2vis = num2vis  # Number of queries to visualize
        for i in range(0, num2vis):
            query_name = self.img_paths[i]  # Simple image name (without prefix)
            query_pid = self.pids[i]  # Query person ID
            query_camid = self.camids[i]  # Query camera ID

            # Get topk ranked indices, considering only different cameras
            ranked_indices = [idx for idx in np.argsort(distmat[i]) if self.camids[idx + self.num_query] != query_camid]
            ranked_indices = ranked_indices[:topk]  # Get topk indices

            # Load query images
            query_imgs = [self.load_image(os.path.join(self.img_prefixes[modality], query_name))
                          for modality in ['RGB', 'NIR', 'TIR']]

            # Load gallery images
            gallery_paths = [self.img_paths[idx + self.num_query] for idx in ranked_indices]  # List of names
            gallery_paths = [[os.path.join(self.img_prefixes[modality], path) for path in gallery_paths]
                             for modality in ['RGB', 'NIR', 'TIR']]

            # Prepare new gallery paths
            new_gallery_paths = []
            for j in range(topk):
                gallery_path = [gallery_paths[0][j], gallery_paths[1][j], gallery_paths[2][j]]
                new_gallery_paths.append(gallery_path)
            gallery_paths = new_gallery_paths

            gallery_pids = [self.pids[idx + self.num_query] for idx in ranked_indices]  # Get corresponding PIDs

            # Load gallery images
            gallery_imgs = [[self.load_image(path) for path in paths] for paths in zip(*gallery_paths)]

            # Plot and save
            self.plot_images(query_imgs, gallery_imgs, gallery_pids, query_pid,
                             save_path=os.path.join(save_dir, f'query_{i}_results.png'))

    def plot_images(self, query_imgs, gallery_imgs, gallery_pids, query_pid, save_path=None):
        print("enter plot_images",end='\r')
        """
        Helper function to plot the query images and the gallery images.
        Correctly retrieved images are marked with a green box, incorrect ones with a red box.
        """
        num_results = len(gallery_imgs[0])  # Number of top-ranked results
        fig, axs = plt.subplots(3, num_results + 1, figsize=(20, 8))

        # Display the query images
        modalities = ['RGB', 'NIR', 'TIR']
        for j, (img, modality) in enumerate(zip(query_imgs, modalities)):
            axs[j, 0].imshow(img)
            axs[j, 0].set_title(f"Query {modality}", fontsize=21)
            axs[j, 0].axis('off')

        # Display the top-ranked results
        for i, (imgs, pid) in enumerate(zip(zip(*gallery_imgs), gallery_pids)):  # Unzip and process each modality
            for j, (img, modality) in enumerate(zip(imgs, modalities)):
                axs[j, i + 1].imshow(img)
                axs[j, i + 1].axis('off')

                # Add a green or red rectangle around the image
                color = 'green' if pid == query_pid else 'red'
                rect = patches.Rectangle((0, 0), img.shape[1], img.shape[0], linewidth=10, edgecolor=color,
                                         facecolor='none')
                axs[j, i + 1].add_patch(rect)
                axs[j, i + 1].set_title(f"Rank {i + 1}", fontsize=22)

        plt.tight_layout()
        if save_path:
            print("save plot_images",end='\r')
            plt.savefig(save_path)
        # plt.show()
        plt.close()

    
    # 1 TSNE
    def showPointMultiModal(self, features, real_label, draw_label, save_path='TSNE/base_A_cls_pat'):
        print("enter T-SNE")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        id_show = 25
        save_path = os.path.join(save_path, str(draw_label) + ".pdf")
        print("Draw points of features to {}".format(save_path))
        indices = find_label_indices(real_label, draw_label, max_indices_per_label=id_show)
        feat = features[indices]
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=1, learning_rate=100, perplexity=60)
        features_tsne = tsne.fit_transform(feat)
        colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a', '#fb9a99',
                  '#fdbf6f', '#cab2d6', '#ffff99']
        MARKS = ['*']
        plt.figure(figsize=(10, 10))
        for i in range(features_tsne.shape[0]):
            plt.scatter(features_tsne[i, 0], features_tsne[i, 1], s=300, color=colors[i // id_show], marker=MARKS[0],
                        alpha=0.4)
        plt.title("t-SNE Visualization of Different IDs")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        # plt.legend()
        plt.savefig(save_path)
        #plt.show()
        plt.close()

    # 3 similarity distribution
    def plot_similarity_distribution(self, features, ids, title="Cosine Similarity Distribution",base_dir = 'Similarity_pic',condition='baseline'):
        """
        绘制同ID和不同ID的余弦相似度分布图，并在图中标注均值。
        字体全部放大。

        参数:
            features (numpy.ndarray): 特征矩阵，形状为 (num_samples, feature_dim)。
            ids (numpy.ndarray): 样本对应的ID，形状为 (num_samples,)。
            title (str): 分布图的标题。
        """
        print("enter plot_similarity_distribution")
        
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        features = features.cpu().detach().numpy()
        # Step 1: 计算余弦相似度矩阵
        similarity_matrix = cosine_similarity(features)

        # Step 2: 筛选正样本对（同ID）和负样本对（不同ID）
        positive_similarities = []
        negative_similarities = []

        num_samples = features.shape[0]
        for i in range(num_samples):
            for j in range(i + 1, num_samples):  # 避免重复计算
                if ids[i] == ids[j]:
                    positive_similarities.append(similarity_matrix[i, j])  # 同ID对
                else:
                    negative_similarities.append(similarity_matrix[i, j])  # 不同ID对

        # 转为数组方便处理
        positive_similarities = np.array(positive_similarities)
        negative_similarities = np.array(negative_similarities)

        # Step 3: 计算均值
        mean_positive = np.mean(positive_similarities)
        mean_negative = np.mean(negative_similarities)

        # Step 4: 绘制分布图
        plt.figure(figsize=(9, 8))  # 放大图像
        sns.kdeplot(positive_similarities, label='Positive Pairs (Same ID)', color='green', shade=True)
        sns.kdeplot(negative_similarities, label='Negative Pairs (Different ID)', color='red', shade=True)
        plt.axvline(mean_positive, color='green', linestyle='--',
                    label=f'Mean Positive Similarity: {mean_positive:.4f}')
        plt.axvline(mean_negative, color='red', linestyle='--', label=f'Mean Negative Similarity: {mean_negative:.4f}')

        # 在图中标注均值
        plt.text(mean_positive, 0.02, f'{mean_positive:.4f}', color='green', fontsize=18, ha='center', va='bottom')
        plt.text(mean_negative, 0.02, f'{mean_negative:.4f}', color='red', fontsize=18, ha='center', va='bottom')

        # 图例和标题（字体放大）
        plt.xlabel('Cosine Similarity', fontsize=22)
        plt.ylabel('Density', fontsize=22)
        plt.title(title, fontsize=26)
        plt.legend(fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.5)
        filename = 'sd_{}.pdf'  # 文件名模板
        save_path = f'{base_dir}/{filename.format(condition)}'
        plt.savefig(save_path)
        # 显示图像
        #plt.show()
        # 获取 KDE 曲线的采样点
        x_vals = np.linspace(min(min(positive_similarities), min(negative_similarities)),
                            max(max(positive_similarities), max(negative_similarities)), 1000)
        
        # 计算正样本和负样本的 KDE 曲线
        from scipy.stats import gaussian_kde
        kde_positive = gaussian_kde(positive_similarities)
        kde_negative = gaussian_kde(negative_similarities)
        
        y_positive = kde_positive(x_vals)
        y_negative = kde_negative(x_vals)

        # 找到两个 KDE 曲线的交点
        intersection_indices = np.where(np.diff(np.sign(y_positive - y_negative)))[0]
        intersection_x = x_vals[intersection_indices]
        intersection_y = y_positive[intersection_indices]  # 或 y_negative[intersection_indices]

        # 如果没有交点，则重叠面积为 0 或 1（取决于哪个分布完全包含另一个）
        if len(intersection_indices) == 0:
            if np.all(y_positive <= y_negative):
                overlap_area = simps(y_positive, x_vals)
            else:
                overlap_area = simps(y_negative, x_vals)
        else:
            # 计算交点之间的面积
            overlap_area = 0.0
            # 遍历所有交点区间
            for i in range(len(intersection_indices)):
                if i == 0:
                    x_start = x_vals[0]
                else:
                    x_start = intersection_x[i - 1]
                
                if i == len(intersection_indices) - 1:
                    x_end = x_vals[-1]
                else:
                    x_end = intersection_x[i + 1]
                
                # 找到当前区间的 x 范围
                mask = (x_vals >= x_start) & (x_vals <= x_end)
                x_segment = x_vals[mask]
                y_positive_segment = y_positive[mask]
                y_negative_segment = y_negative[mask]
                
                # 取两个 KDE 曲线的较小值进行积分
                overlap_segment = np.minimum(y_positive_segment, y_negative_segment)
                overlap_area += simps(overlap_segment, x_segment)

        print(f"Overlap Area between Positive and Negative Distributions: {overlap_area:.4f}")


        # Step 5: 打印统计信息
        print(f"Mean Positive Similarity: {mean_positive:.4f}")
        print(f"Mean Negative Similarity: {mean_negative:.4f}")
        print(f"Positive Similarity Std Dev: {np.std(positive_similarities):.4f}")
        print(f"Negative Similarity Std Dev: {np.std(negative_similarities):.4f}")
    


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()


def find_label_indices(label_list, target_labels, max_indices_per_label=1):
    indices = []
    counts = {label: 0 for label in target_labels}
    for index, label in enumerate(label_list):
        if label in target_labels and counts[label] < max_indices_per_label:
            indices.append(index)
            counts[label] += 1
    sorted_indices = sorted(indices, key=lambda index: (label_list[index], index))
    return sorted_indices


def _calculate_similarity(pre_fusion_tokens, post_fusion_tokens):
    """
    计算融合前后patch token的相似度

    Args:
        pre_fusion_tokens: 融合前patch token
        post_fusion_tokens: 融合后patch token

    Returns:
        similarities: 融合前后patch token的相似度
    """

    # 计算余弦相似度
    similarities = torch.nn.functional.cosine_similarity(pre_fusion_tokens, post_fusion_tokens,
                                                         dim=-1).cpu().detach().numpy()

    # # 将相似度平均到每个patch
    # similarities = torch.mean(similarities, dim=1).squeeze().cpu().detach().numpy()

    return similarities


def visualize_similarity(pre_fusion_src_tokens, pre_fusion_tgt_tokens, post_fusion_src_tokens, post_fusion_tgt_tokens,
                         writer=None, epoch=None, mode=1,
                         pattern=None, figure_size=(6, 6), seaborn_style='whitegrid'):
    """
    可视化融合前后patch token的相似度分布

    Args:
        pre_fusion_src_tokens: 融合前源图像patch token
        pre_fusion_tgt_tokens: 融合前目标图像patch token
        post_fusion_src_tokens: 融合后源图像patch token
        post_fusion_tgt_tokens: 融合后目标图像patch token
        writer: tensorboardX SummaryWriter
        epoch: epoch
        mode: 模式，1代表源图像，2代表目标图像
        pattern: 融合模式，r2t, r2n, n2t, n2r, t2r, t2n
        figure_size: 图像大小
        seaborn_style: seaborn风格

    Returns:
        None
    """

    # 计算融合前后patch token的相似度
    similarities_ori = _calculate_similarity(pre_fusion_src_tokens, pre_fusion_tgt_tokens)
    similarities = _calculate_similarity(post_fusion_src_tokens, post_fusion_tgt_tokens)

    # 设置seaborn风格
    sns.set(style=seaborn_style)

    # 创建画图对象
    fig, ax = plt.subplots(figsize=figure_size)

    # 绘制融合前后相似度分布图
    sns.kdeplot(similarities, color='b', label='Before ..', ax=ax, multiple="stack")
    sns.kdeplot(similarities_ori, color='g', label='After ..', ax=ax, multiple="stack")

    # 设置标题和标签
    if pattern == 'r2t':
        sign = 'R and T'
    elif pattern == 'r2n':
        sign = 'R and N'
    elif pattern == 'n2t':
        sign = 'N and T'
    elif pattern == 'n2r':
        sign = 'N and R'
    elif pattern == 't2r':
        sign = 'T and R'
    elif pattern == 't2n':
        sign = 'T and N'
    plt.title("Similarity Distribution between {}".format(sign), fontsize=18, fontweight='bold')
    plt.xlabel("Cosine Similarity", fontsize=16, fontweight='bold')
    plt.ylabel("Density", fontsize=16, fontweight='bold')
    # 设置x轴刻度标签大小
    plt.xticks(fontsize=15)

    # 设置y轴刻度标签大小
    plt.yticks(fontsize=15)
    # 添加图例
    plt.legend(loc='upper right', fontsize=17)

    # 显示图像
    plt.show()

    # 将图像写入tensorboard
    if writer is not None:
        writer.add_figure('Similarity_{}'.format(sign), plt.gcf(), epoch)