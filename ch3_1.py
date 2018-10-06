import torch as t
import torch



class OurModule(t.nn.Module):

    def __init__(self, num_inputs, num_classes, dropout_prob=0.3):

        super(OurModule, self).__init__()

        self.pipe = t.nn.Sequential(t.nn.Linear(num_inputs, 5), t.nn.ReLU(),
                                    t.nn.Linear(5, 20), t.nn.ReLU(), t.nn.Linear(20, num_classes),
                                    t.nn.Dropout(dropout_prob), t.nn.Softmax(dim=1))

    def forward(self, x):

        return self.pipe(x)


if __name__ == "__main__":

    net = OurModule(2, 3)
    v = torch.FloatTensor([[2, 3]])
    out = net(v)
    print(out)


################### COMMON BLUEPRINT OF A TRAINING LOOP ##############################


# for batch_samples, batch_labels in iterate_batches(data, batch_size = 32):
#
#     batch_samples_t = t.Tensor(batch_samples)
#     batch_labels_t = t.Tensor(batch_labels)
#     out_t = net(batch_labels_t)
#     loss_t = loss_function(out_t, batch_labels_t)
#     loss_t.backward()
#     optimizer.step()
#     optimizer.zero_grad()
