---
layout: post
title:  "Configmaps"
date:   2022-01-06 12:44:21 +0200
categories: k8s
comments: true
author: me
---
Recently I was working with configmaps to make dynamic configuration changes that would then affect the pod at runtime without restarting it. For this purpose, I mounted the configmap as volume and used a filewatcher to register changes. It was then when I noticed that k8s configmaps store their data in an interesting fashion.

## Symlinks galore
Actually, what I am about to describe does not just apply to configmaps, but to volume mounts in general. But let's just use the following configmap as example.
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-config
  namespace: default
data:
  myConfigmapKey: "myValue"
```
Listing the contents of the volume mount path inside the pod shows two symlinks and one directory, where a file named `myConfigmapKey` is stored, and the content of that file is `myValue`.
```bash
drwxr-xr-x Jan  6 09:40 ..2022_01_06_09_40_46.156256083
lrwxrwxrwx Jan  6 09:40 ..data -> ..2022_01_06_09_40_46.156256083
lrwxrwxrwx Jan  6 09:15 myConfigmapKey -> ..data/myConfigmapKey
```
When I update the configmap with `kubectl apply -f`, what I eventually see in the pod is
```bash
drwxr-xr-x Jan  6 09:58 ..2022_01_06_09_58_07.583983766
lrwxrwxrwx Jan  6 09:58 ..data -> ..2022_01_06_09_58_07.583983766
lrwxrwxrwx Jan  6 09:15  myConfigmapKey -> ..data/myConfigmapKey
```
But there is more: The filewatcher logs showed that during the upgrade, a symlink called `data_tmp` is used and deleted again after the update is complete. 

A later [search][1] confirms the order of events:
1. upgrade is triggered
2. new timestamp folder is created with the updated configmap content
3. a new symlink `data_tmp` pointing to the new timestamp folder is created
4. `data_tmp` is renamed to `data`
5. the old timestamp folder is removed

Why use symlinks? The advantage to an in-place modification of a file is that the pod can continue to access the old, uncompromised configmap value while the new content is still written in the new timestamp folder. The switch which happens in step 4 is atomic, so there will never be some awkward in-between data being read. Pretty cool!

My only remaining question is: Are 2 symlinks really required? Would it suffice to just have 1 symlink, skipping `data`/`data_tmp` and directly using `myConfigmapKey`/`myConfigmapMapKey_tmp`?
```
drwxr-xr-x Jan  6 09:58 ..2022_01_06_09_58_07.583983766
lrwxrwxrwx Jan  6 09:58 myConfigmapKey -> ..2022_01_06_09_58_07.583983766/myConfigmapKey
```

[1]: https://pkg.go.dev/k8s.io/kubernetes/pkg/volume/util#AtomicWriter.Write
