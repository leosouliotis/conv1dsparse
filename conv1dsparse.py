class Conv1dsparse(nn.Module):

    def __init__(self, conv, lengths):
        super(Conv1dsparse, self).__init__()

        self.kernel_size = conv.kernel_size[0]
        self.out_channels = conv.out_channels
        self.lengths = lengths
        self.weight = conv.weight

        if conv.bias is None:
            self.bias_ind = False
        else:
            self.bias = conv.bias
            self.bias_ind = True

    def forward(self, sparse_tensor):
        # 3D tensor storing the position of each non-negative element of each datum
        indices = sparse_tensor._indices()
        N = sparse_tensor.shape[0]
        #if not self.lengths:
        #    self.lengths = [sparse_tensor.shape[2] for _ in range(N)]
        dims = self.weight.shape
        out_channels = dims[0]  # No of output channels
        out_len = sparse_tensor.size(2) - self.kernel_size + 1
        out_ch = torch.zeros([N, out_channels, out_len])
        prev_gen = 0

        for datum in range(N):  # looping for all genomes

            genome_pos = [i for i in range(prev_gen, prev_gen + self.lengths[datum])]
            prev_gen += self.lengths[datum]

            # Dictionary for faster iteration
            keggs_present = indices[1][genome_pos].int().tolist()
            position_tensor = indices[2][genome_pos].int().tolist()
            keggs_present_dict = defaultdict(list)
            for k, v in enumerate(keggs_present):
                keggs_present_dict[v].append(position_tensor[k])

            adj_list = [2 if len(v) > 1 and any([v[i + 1] - v[i] < self.kernel_size for i in range(len(v) - 1)])
                        else 1
                        if len(v) > 1 and all([v[i + 1] - v[i] >= self.kernel_size for i in range(len(v) - 1)])
                        else 0
                        for k, v in keggs_present_dict.items()]

            # For each input chanel present
            for ind, values in enumerate(keggs_present_dict.items()):
                in_channel, position = values
                if adj_list[ind] == 0 or adj_list[ind] == 1:

                    for el in position:
                        filter_range = [[i + j for j in range(self.kernel_size)]
                                        for i in range(el - self.kernel_size + 1, el + 1) if out_len > i >= 0]
                        kernel_pos = [[k for k, l in enumerate(i) if l == el] for i in filter_range]
                        temp = [i[0] for i in filter_range]
                        convolutions = torch.stack([self.weight[:, in_channel, i].view(out_channels, 1)
                                                    for i in kernel_pos], 1).view(out_channels, len(temp))
                        out_ch[datum, :, temp] += convolutions
                else:

                    cont = sorted(list(set(sum([[k, l] for l, i in enumerate(position)
                                                for k, j in enumerate(position)
                                                if (i != j and i < j and abs(i - j) < self.kernel_size)], []))))

                    non_cont = [k for k in range(len(position)) if k not in cont]

                    if cont:
                        position_cont = [position[el] for el in cont]
                        split_points = [k + 1 for k, i in enumerate(range(len(position_cont) - 1))
                                        if position_cont[i + 1] - position_cont[i] > self.kernel_size]

                        if not split_points: split_points = [max(cont) + 1]

                        splits = [i for i in np.split(cont, split_points) if len(i) > 1]
                        for el in splits:
                            el = [position[i] for i in el.tolist()]
                            min_el, max_el, len_el = min(el), max(el), len(el)
                            filter_range = [[i + j for j in range(self.kernel_size)] for i in
                                            range(min_el - self.kernel_size + 1, max_el + 1) if out_len > i >= 0]

                            kernel_pos = [[k for k, l in enumerate(i) if l in el] for i in filter_range]
                            temp = [i[0] if len(i) > 0 else i for i in filter_range]
                            convolutions = torch.stack(
                                [self.weight[:, in_channel, i].sum(1).view(out_channels, 1) if len(i) > 1
                                 else self.weight[:, in_channel, i].view(out_channels, 1)
                                 for i in kernel_pos], 1).view(out_channels, len(temp))
                            out_ch[datum, :, temp] += convolutions

                    if non_cont:

                        for el in non_cont:
                            el = position[el]
                            filter_range = [[i + j for j in range(self.kernel_size)]
                                            for i in range(el - self.kernel_size + 1, el + 1) if out_len > i >= 0]
                            kernel_pos = [[k for k, l in enumerate(i) if l == el] for i in filter_range]
                            temp = [i[0] for i in filter_range]
                            convolutions = torch.stack([self.weight[:, in_channel, i].view(out_channels, 1)
                                                        for i in kernel_pos], 1).view(out_channels, len(temp))
                            out_ch[datum, :, temp] += convolutions

        if self.bias_ind:
            out_ch += torch.tensor([[el for el in self.bias for _ in range(out_len)]
                                    for _ in range(N)]).view(N, self.out_channels, out_len)

        return out_ch
