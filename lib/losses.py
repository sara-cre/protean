import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F



class ConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ConLoss, self).__init__()
        self.temperature = temperature
        print("Temperature: ", temperature)
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, global_protos=None, mask=None):
        """Compute contrastive loss between feature and global prototype
        """


        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        #print("features shape: ", features.shape)
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)


        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        #print(contrast_feature.shape)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            # anchor_feature = contrast_feature
            anchor_count = contrast_count
            anchor_feature = torch.zeros_like(contrast_feature)
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))


        # generate anchor_feature
        for i in range(batch_size*anchor_count):
            anchor_feature[i, :] = global_protos[labels[i%batch_size].item()][0]

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        if torch.isnan(logits).any():
            raise ValueError('NaN values detected in logits')

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        """if (mask.sum(1) == 0).any():
            raise ValueError('Zero division error: mask.sum(1) has zero values')"""


        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        if torch.isnan(log_prob).any():
            raise ValueError('NaN values detected in log_prob')

        # compute mean of log-likelihood over positive
        mask_sum = mask.sum(1)
        if (mask_sum == 0).any():
            mask_sum[mask_sum == 0] = 1  # Avoid division by zero

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        #mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        if torch.isnan(mean_log_prob_pos).any():
            raise ValueError('NaN values detected in mean_log_prob_pos')

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class ConLoss__(nn.Module):
    def __init__(self, temperature=0.07):
        super(ConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels, global_protos, local_protos=None):
        batch_size = features.size(0)
        print(global_protos)
        # Global prototype-based loss L_g
        global_loss = self.compute_loss(features, labels, global_protos)
        

        # Local prototype-based loss L_p
        local_loss = 0.0
        if local_protos is not None:
            m = len(local_protos)  # Number of clients
            for p in range(m):
                local_loss += self.compute_loss(features, labels, local_protos[p])
            local_loss /= m  # Normalize by the number of clients

        # Combine the global and local losses
        total_loss = global_loss + local_loss
        return total_loss

    def compute_loss(self, features, labels, prototypes):
        loss = 0.0
        batch_size = features.size(0)

        # Prepare the positive prototypes
        pos_protos = torch.stack([prototypes[label.item()][0] for label in labels]).to(features.device)

        # Compute the dot product between features and positive prototypes
        pos_sim = torch.einsum('nc,nc->n', [features, pos_protos]) / self.temperature

        # Prepare the negative prototypes
        neg_protos = []
        for l in prototypes:
            if l != labels.item():
                neg_proto = prototypes[l][0].to(features.device)
                neg_protos.append(neg_proto)
        neg_protos = torch.stack(neg_protos)

        # Compute the dot product between features and negative prototypes
        neg_sim = torch.matmul(features, neg_protos.t()) / self.temperature

        # Exclude positive prototypes from negative prototypes (by setting mask)
        mask = torch.ones_like(neg_sim)
        mask.scatter_(1, labels.unsqueeze(1), 0)
        neg_sim = neg_sim * mask

        # Compute the log-sum-exp for denominator
        denominator = torch.logsumexp(neg_sim, dim=1)

        # Compute the global contrastive loss
        loss = -pos_sim + denominator
        loss = loss.mean()

        return loss



class ConLoss_(nn.Module):
    def __init__(self, temperature=0.07):
        super(ConLoss, self).__init__()
        self.temperature = 0.7 #temperature

    def forward2(self, features, labels, global_protos, local_protos=None):
        # Initialize loss components
        global_loss = 0.0
        local_loss = 0.0

        batch_size = features.size(0)

        for i in range(batch_size):
            z_x = features[i]
            label = labels[i].item()

            # Positive prototype from the global prototypes (average if it's a list)
            pos_proto_list = global_protos[label]
            if not isinstance(pos_proto_list, list):
                pos_proto_list = [pos_proto_list]  # Ensure it's a list
            pos_proto = torch.stack(pos_proto_list).mean(dim=0).to(features.device)

            # Negative prototypes from the global prototypes (average if they're lists)
            neg_protos = []
            for l in global_protos:
                if l != label:
                    proto_list = global_protos[l]
                    if not isinstance(proto_list, list):
                        proto_list = [proto_list]  # Ensure it's a list
                    proto = torch.stack(proto_list).mean(dim=0).to(features.device)
                    neg_protos.append(proto)

            if len(neg_protos) > 0:
                neg_protos = torch.stack(neg_protos)

                # Compute the global contrastive loss Lg
                numerator = torch.exp(torch.dot(z_x, pos_proto) / self.temperature)
                denominator = torch.sum(torch.exp(torch.matmul(z_x.unsqueeze(0), neg_protos.t()) / self.temperature))
                global_loss += -torch.log(numerator / denominator)
            else:
                global_loss += 0  # Handle case where there are no negative prototypes

        global_loss /= batch_size  # Average over batch

        # Compute the local prototype-based loss L_p if local prototypes are provided
        if local_protos is not None:
            m = len(local_protos)  # Number of local prototypes (clients)

            for i in range(batch_size):
                z_x = features[i]
                label = labels[i].item()

                for p in range(m):
                    if label in local_protos[p]:
                        # Positive prototype for client p
                        pos_proto_list = local_protos[p][label] 
                        if not isinstance(pos_proto_list, list):
                            pos_proto_list = [pos_proto_list]  # Ensure it's a list
                        pos_proto = torch.stack(pos_proto_list).mean(dim=0).to(features.device)

                        neg_protos = []
                        for l in local_protos[p]:
                            if l != label:
                                proto_list = local_protos[p][l]
                                if not isinstance(proto_list, list):
                                    proto_list = [proto_list]  # Ensure it's a list
                                proto = torch.stack(proto_list).mean(dim=0).to(features.device)
                                neg_protos.append(proto)

                        if len(neg_protos) > 0:
                            neg_protos = torch.stack(neg_protos)
                            # Calculate the local contrastive loss L_p
                            numerator = torch.exp(torch.dot(z_x, pos_proto) / self.temperature)
                            denominator = torch.sum(torch.exp(torch.matmul(z_x.unsqueeze(0), neg_protos.t()) / self.temperature))
                            local_loss += -torch.log(numerator / denominator)

            local_loss /= (batch_size * m)  # Normalize by batch size and number of clients

        # Combine the global and local losses
        
        #print("global_loss", global_loss)
        #print("local_loss", local_loss)
        total_loss = global_loss + local_loss

        return  total_loss

    def forward(self, features, labels, global_protos, local_protos=None):
        lg = self.global_prototype_loss(features, global_protos, labels, self.temperature)
        #lp = self.local_prototype_loss(features, local_protos, labels, self.temperature) if local_protos is not None else 0.0
        return lg  #+ lp

    def global_prototype_loss(self, z, C_dict, y, tau):
        """
        Computes the global prototype-based loss L_g.

        Parameters:
        z (torch.Tensor): The fused representation z(x), shape (batch_size, feature_dim)
        C_dict (dict): A dictionary of global prototypes, where each key is a class label 
                    and each value is a list containing one tensor of shape (feature_dim,)
        y (torch.Tensor): The ground-truth labels y, shape (batch_size,)
        tau (float): The temperature parameter

        Returns:
        torch.Tensor: The global prototype-based loss L_g
        """
        batch_size = z.size(0)
        
        # Initialize the logits tensor
        logits = torch.zeros((batch_size, len(C_dict)), device=z.device)

        for i, class_label in enumerate(C_dict.keys()):
            # Extract the prototype for the current class
            prototype = C_dict[class_label][0]
            # Ensure the prototype is a tensor and has at least one dimension
            if prototype.dim() == 0:
                prototype = prototype.unsqueeze(0)
            # Compute the logit for this prototype
            logits[:, i] = torch.matmul(z, prototype) / tau

        # Extract the logits for the correct class
        correct_class_logits = logits[torch.arange(batch_size), y]

        # Compute the mask for incorrect classes
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(batch_size), y] = False

        # Compute the loss
        loss = -correct_class_logits + torch.logsumexp(logits.masked_select(mask).view(batch_size, len(C_dict) - 1), dim=1)
        
        return loss.mean()

    def local_prototype_loss(self, z, C_p_dict, y, tau):
        num_clients = len(C_p_dict)
        total_loss = 0.0

        for i in range(num_clients):
            C_dict = C_p_dict[i]
            lg = self.global_prototype_loss(z, C_dict, y, tau)
            total_loss += lg / num_clients
        
        return total_loss
    def local_prototype_loss_(self, z, C_p_dict, y, tau):
        """
        Computes the local prototype-based loss L_p.

        Parameters:
        z (torch.Tensor): The fused representation z(x), shape (batch_size, feature_dim)
        C_p_dict (dict): A dictionary where keys are class labels and values are lists of tensors (prototypes) of shape (feature_dim,)
        y (torch.Tensor): The ground-truth labels y, shape (batch_size,)
        tau (float): The temperature parameter

        Returns:
        torch.Tensor: The local prototype-based loss L_p
        """
        batch_size = z.size(0)
        num_classes = len(C_p_dict)
        m = len(next(iter(C_p_dict.values())))  # Number of prototypes per class

        # Initialize the logits tensor
        logits = torch.zeros((batch_size, m, num_classes), device=z.device)

        for i, class_label in enumerate(C_p_dict.keys()):
            # Extract the prototypes for the current class
            prototypes = C_p_dict[class_label]
            for j in range(m):
                # If prototypes[j] is a list, convert it to a tensor
                prototype_tensor = torch.stack(prototypes[j]) if isinstance(prototypes[j], list) else prototypes[j]
                # Reshape prototype_tensor if necessary
                if prototype_tensor.dim() == 1:
                    prototype_tensor = prototype_tensor.unsqueeze(0)  # Make it (1, feature_dim)
                # Compute the logit for each prototype in the list
                logits[:, j, i] = torch.matmul(z, prototype_tensor.T) / tau

        # Compute the loss for the correct class
        correct_class_logits = logits[:, :, y].mean(dim=1)

        # Compute the loss
        loss = -correct_class_logits.mean(dim=1) + torch.logsumexp(logits, dim=2).mean(dim=1)
        
        return loss.mean()



class ConLoss_op(nn.Module):
    def __init__(self, temperature=0.07):
        super(ConLoss_op, self).__init__()
        self.temperature = temperature  # Use the correct temperature value

    def _get_prototypes(self, proto_dict, label):
        """Helper function to get positive and negative prototypes."""
        pos_proto_list = proto_dict[label]
        if not isinstance(pos_proto_list, list):
            pos_proto_list = [pos_proto_list]  # Ensure it's a list
        pos_proto = torch.stack(pos_proto_list).mean(dim=0) if len(pos_proto_list) > 1 else pos_proto_list[0]

        neg_protos = []
        for l in proto_dict:
            if l != label:
                proto_list = proto_dict[l]
                if not isinstance(proto_list, list):
                    proto_list = [proto_list]  # Ensure it's a list
                if len(proto_list) > 0:
                    proto = torch.stack(proto_list).mean(dim=0) if len(proto_list) > 1 else proto_list[0]
                    neg_protos.append(proto)

        return pos_proto, neg_protos
    def _contrastive_loss(self, features, pos_proto, neg_protos):
        """Compute contrastive loss given features, positive and negative prototypes."""
        if len(neg_protos) == 0:
            return 0

        neg_protos = torch.stack(neg_protos)
        numerator = torch.exp(torch.dot(features, pos_proto) / self.temperature)
        denominator = torch.sum(torch.exp(torch.matmul(features.unsqueeze(0), neg_protos.t()) / self.temperature))
        
        return -torch.log(numerator / denominator)

    def forward(self, features, labels, global_protos, local_protos=None):
        batch_size = features.size(0)

        # Compute global loss
        global_loss = 0.0
        for i in range(batch_size):
            z_x = features[i]
            label = labels[i].item()
            pos_proto, neg_protos = self._get_prototypes(global_protos, label)
            global_loss += self._contrastive_loss(z_x, pos_proto.to(features.device), neg_protos)

        global_loss /= batch_size  # Average over batch

        # Compute local loss if local_protos are provided
        local_loss = 0.0
        if local_protos is not None:
            m = len(local_protos)  # Number of local prototypes (clients)
            for i in range(batch_size):
                z_x = features[i]
                label = labels[i].item()
                for p in range(m):
                    pos_proto, neg_protos = self._get_prototypes(local_protos[p], label)
                    local_loss += self._contrastive_loss(z_x, pos_proto.to(features.device), neg_protos)

            local_loss /= (batch_size * m)  # Normalize by batch size and number of clients

        total_loss = global_loss + local_loss
        return total_loss


import torch
import torch.nn.functional as F

class OptimizedLoss(nn.Module):
    def __init__(self, temperature=0.07, num_clients=1):
        super(OptimizedLoss, self).__init__()
        self.temperature = temperature
        self.num_clients = num_clients

    def forward(self, features, labels, global_protos, local_protos=None):
        # Normalize features and prototypes
        features = F.normalize(features, dim=1)
        global_protos = {k: F.normalize(torch.stack(v), dim=1).mean(dim=0) for k, v in global_protos.items()}
        
        if local_protos is not None:
            local_protos = [{k: F.normalize(torch.stack(v), dim=1).mean(dim=0) for k, v in local.items()} for local in local_protos]

        # Calculate L_g
        global_loss = self.compute_loss(features, labels, global_protos)

        # Calculate L_p if local_protos are provided
        if local_protos is not None:
            local_loss = 0.0
            for p in range(self.num_clients):
                local_loss += self.compute_loss(features, labels, local_protos[p])
            local_loss /= self.num_clients
        else:
            local_loss = 0.0

        return global_loss + local_loss

    def compute_loss(self, features, labels, protos):
        batch_size = features.size(0)
        device = features.device

        # Calculate similarity matrix
        logits = torch.matmul(features, torch.stack(list(protos.values())).T) / self.temperature

        # Create labels for the cross-entropy loss
        target_logits = torch.tensor([protos[l.item()] for l in labels]).to(device)
        labels_idx = torch.tensor([list(protos.keys()).index(l.item()) for l in labels]).to(device)

        # Apply cross-entropy loss
        loss = F.cross_entropy(logits, labels_idx)

        return loss



class ConLoss2(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ConLoss2, self).__init__()
        self.temperature = temperature
        print("Temperature: ", temperature)
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, global_protos=None, mask=None):
        """Compute contrastive loss between feature and global prototype
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        #print("features shape: ", features.shape)

        
        batch_size = features.shape[0]
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        #print(contrast_feature.shape)
        anchor_feature = features

        # generate anchor_feature
        for i in range(batch_size):
            anchor_feature[i] = global_protos[labels[i].item()]

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
