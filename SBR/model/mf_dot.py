import torch

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD


class MatrixFactorizatoinDotProduct(torch.nn.Module):
    def __init__(self, config, n_users, n_items, use_item_bias=False, use_user_bias=False):
        super(MatrixFactorizatoinDotProduct, self).__init__()

        self.user_embedding = torch.nn.Embedding(n_users, config["embedding_dim"])
        self.item_embedding = torch.nn.Embedding(n_items, config["embedding_dim"])

        self.use_item_bias = use_item_bias
        self.use_user_bias = use_user_bias

        if self.use_user_bias:
            self.user_bias = torch.nn.Parameter(torch.zeros(n_users))
        if self.use_item_bias:
            self.item_bias = torch.nn.Parameter(torch.zeros(n_items))

    def forward(self, batch):
        users = batch[INTERNAL_USER_ID_FIELD].squeeze()
        items = batch[INTERNAL_ITEM_ID_FIELD].squeeze()

        user_embeds = self.user_embedding(users)
        item_embeds = self.item_embedding(items)

        # we want to multiply each user with its corresponding item:  # todo which is faster? time these? very similar not much different
        # 1: elementwise multiplication of user and item embeds and them sum on dim1
        output = torch.sum(torch.mul(user_embeds, item_embeds), dim=1)
        # 2: taking the diagonal of matrixmul of user and item embeds:
        #output = torch.diag(torch.matmul(user_embeds, item_embeds.T))
        if self.use_item_bias:
            output = output + self.item_bias[items]
        if self.use_user_bias:
            output = output + self.user_bias[users]
        return output.unsqueeze(1)  # do not apply sigmoid and use BCEWithLogitsLoss
