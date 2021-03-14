---
layout: post
title:  "Diving Into Controller Runtime"
date:   2021-03-12 17:16:30 +0200
categories: k8s
comments: true
excerpt_separator: <!--more-->
---
In my previous post [Writing Controllers for Kubernetes Custom Resources]() I explored the development process using pure client-go and how it differs using controller-runtime and kubebuilder. In this post, I explore how controller-runtime makes use of client-go under the hood.

<!--more-->
As described in my previous post, to develop an operator with controller-runtime, we need to
1. define CRD types
2. generate deepcopy functions using `deepcopy-gen`
3. write reconciler logic
4. create and run the controller
5. deployment-related configuration steps (out of cope for this post)
Let's briefly walk through the steps using code from a project generated with kubebuilder. You can find the [full source code in this repo](https://github.com/ViviLearns2Code/myoperator/)

```golang
/* snipped source code from https://github.com/ViviLearns2Code/myoperator/blob/main/api/v1/mykind_types.go */
package v1
import (
  "k8s.io/apimachinery/pkg/runtime/schema"
  "sigs.k8s.io/controller-runtime/pkg/scheme"
)

var (
  GroupVersion = schema.GroupVersion{Group: "mygroup.mydomain", Version: "v1"}
  SchemeBuilder = &scheme.Builder{GroupVersion: GroupVersion}
  AddToScheme = SchemeBuilder.AddToScheme
)

type MyKindSpec struct { /* ... */ }
type MyKindStatus struct { /* ... */ }
type MyKind struct { /* ... */ }
type MyKindList struct { /* ... */ }

SchemeBuilder.Register(&MyKind{}, &MyKindList{})
```
After defining the golang types for our CRD, we need to implement the `Reconciler` interface defined by the controller-runtime library
```golang
/* source code from https://github.com/kubernetes-sigs/controller-runtime/blob/master/pkg/reconcile/reconcile.go */
type Reconciler interface {
  Reconcile(context.Context, Request) (Result, error)
}
```
```golang
/* snipped source code from https://github.com/ViviLearns2Code/myoperator/blob/main/controllers/mykind_controller.go */
package controllers
import (
  "k8s.io/apimachinery/pkg/runtime"
  ctrl "sigs.k8s.io/controller-runtime"
  "sigs.k8s.io/controller-runtime/pkg/client"
)

type MyKindReconciler struct {
  client.Client
  Scheme *runtime.Scheme
}

func (r *MyKindReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
  /* snipped business logic */
  return ctrl.Result{}, nil
}
```
If you do not use kubebuilder, you need to generate a file with deepcopy functions for your resource manually using [`deepcopy-gen`](https://github.com/kubernetes/code-generator). The result is the file `api/v1/zz_generated_deepcopy.go`(https://github.com/ViviLearns2Code/myoperator/blob/main/api/v1/zz_generated.deepcopy.go).

And now it is time to use controller-runtime.

## Starting with the Manager
A manager is required to create and start up a controller. The manager needs a kubeconfig (controller-runtime offers `GetConfigOrDie()` to read it from the system) and can be provided with many options. Two examples are `Scheme`, which contains predefined k8s types and our CRD type, or the `Port` used to serve webhooks on. After creating and configuring the manager, we can add our CRD controller to it and start the manager.
```golang
/* snipped source code from https://github.com/ViviLearns2Code/myoperator/blob/main/main.go */
package main

import (
  "k8s.io/apimachinery/pkg/runtime"
  clientgoscheme "k8s.io/client-go/kubernetes/scheme"
  ctrl "sigs.k8s.io/controller-runtime"

  mygroupv1 "myoperator/api/v1" //our CRD schema definitions
  "myoperator/controllers" //our CRD controller implementation
)

var (
  scheme   = runtime.NewScheme()
)

func init() {
  _ = clientgoscheme.AddToScheme(scheme)
  _ = mygroupv1.AddToScheme(scheme)
}

func main() {
  // create manager
  mgr, _ := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
    Scheme:             scheme,
    Port:               9443,
  })
  // create and add controller
  r := &controllers.MyKindReconciler{
    Client: mgr.GetClient(),
    Scheme: mgr.GetScheme(),
  }
  _ = ctrl.NewControllerManagedBy(mgr).
    For(&mygroupv1.MyKind{}).
    Complete(r)
  // start manager
  mgr.Start(ctrl.SetupSignalHandler())
```
With this, all the steps to be done on your side are completed. But how does controller-runtime connect your code to client-go?

## A Manager's Components
A manager is made up of several configurable components and two of its major components are `client` and `cache`. The former is responsible for read and write operations in general, the latter can be used to read data cached in indices to reduce load on the k8s apiserver. It is therefore not surprising that the client component reuses the cache component. 

```golang
import (
  "sigs.k8s.io/controller-runtime/pkg/cache"
  "sigs.k8s.io/controller-runtime/pkg/client"
)
type controllerManager struct {
  /* snipped source code from pkg/manager/internal.go */
  cache cache.Cache
  client client.Client
  // ...
}
```
To understand how these two are linked to client-go, let us take a look at the cache and client constructors, `NewClientBuilder` and `cache.New`.

```golang
import (
  "sigs.k8s.io/controller-runtime/pkg/cache"
)
func setOptionsDefaults(options Options) Options {
  /* snipped source code from controller-runtime, pkg/manager/manager.go */
  if options.ClientBuilder == nil {
    options.ClientBuilder = NewClientBuilder()
  }
  if options.NewCache == nil {
    options.NewCache = cache.New
  }
  return options
}
```

## The Cache
The struct returned by `cache.New` implements the composite `Cache` interface
```golang
type Cache interface {
  /* snipped source code from pkg/cache/cache.go */
  client.Reader
  Informers
}
type Informers interface {
  /* snipped source code from pkg/cache/cache.go */
  GetInformer(ctx context.Context, obj client.Object) (Informer, error)
  GetInformerForKind(ctx context.Context, gvk schema.GroupVersionKind) (Informer, error)
  Start(ctx context.Context) error
  WaitForCacheSync(ctx context.Context) bool
  client.FieldIndexer
}
```
```golang
type Reader interface {
  /* snipped source code from pkg/client/interfaces.go */
  Get(ctx context.Context, key ObjectKey, obj Object) error
  List(ctx context.Context, list ObjectList, opts ...ListOption) error
}
type FieldIndexer interface {
  /* snipped source code from pkg/client/interfaces.go */
  IndexField(ctx context.Context, obj Object, field string, extractValue IndexerFunc) error
}
```
The definition of that struct is spread across several files (`pkg/cache/informer_cache.go`, `pkg/cache/internal/deleg_map.go`, `pkg/cache/internal/informers_map.go`, `pkg/cache/internal/cache_reader.go`). If you strip away the details like support for (un)structured types or multiple resources, the struct is more or less a map that maps from objects or GVKs to client-go `SharedIndexInformer` and `Indexer` objects. The following table shows how the cache uses client-go.

| Functions | Description | Usage of client-go|
| ------------- |:-------------|:-----|
|`Get` | cached read access | reads from the informer's index that is obtained with `cache.SharedIndexInformer.GetIndexer()` |
|`List` | cached read access | same as above |
|`GetInformer` | retrieves informer for a given runtime object (creates one if doesn't exist) |returns  `cache.SharedIndexInformer` |
|`GetInformerForKind`| same as above, but for GKV | same as above |
|`Start`| runs all informers, meaning it will list & watch the k8s apiserver for resource updates | `cache.ListWatch` |
|`WaitForCacheSync`| waits for all caches to sync | `cache.WaitForCacheSync` |
|`IndexField` | adds field indices over the cache | `cache.SharedIndexInformer.AddIndexers` |

The functions `GetInformer` and `GetInformerForKind` return a struct implementing the interface `Informer`, which offers the functions `AddEventHandler`, `AddEventHandlerWithResyncPeriod`, `AddIndexers` , `HasSynced`. The implementation relies on the eponymous functions offered by client-go's `cache.SharedIndexInformer` and can for example be used to register a controller's workqueue to receive updates from the informer.

## The Client
`NewClientBuilder` returns a builder that builds a client satisfying
```golang
type Client interface {
  /* snipped source code from pkg/client/interfaces.go */
  Reader
  Writer
  StatusClient
  Scheme() *runtime.Scheme
  RESTMapper() meta.RESTMapper
}
type Reader interface {
  /* snipped source code from pkg/client/interfaces.go */
  Get(ctx context.Context, key ObjectKey, obj Object) error
  List(ctx context.Context, list ObjectList, opts ...ListOption) error
}
type Writer interface {
  /* snipped source code from pkg/client/interfaces.go */
  Create(ctx context.Context, obj Object, opts ...CreateOption) error
  Delete(ctx context.Context, obj Object, opts ...DeleteOption) error
  Update(ctx context.Context, obj Object, opts ...UpdateOption) error
  Patch(ctx context.Context, obj Object, patch Patch, opts ...PatchOption) error
  DeleteAllOf(ctx context.Context, obj Object, opts ...DeleteAllOfOption) error
}
type StatusClient interface {
 /* snipped source code from pkg/client/interfaces.go */
  Status() StatusWriter
}
```
(Note: `StatusClient` is in charge of updating the status subresource)

The special thing about the returned client is that it will access the k8s apiserver for write operations and the cache for read operations. If the user wants to bypass the cache for read operations, they can configure the builder by setting the `ClientDisableCacheFor` option of the manager. 

```golang
import (
  "sigs.k8s.io/controller-runtime/pkg/cache"
  "sigs.k8s.io/controller-runtime/pkg/client"
)
func (n *newClientBuilder) Build(cache cache.Cache, config *rest.Config, options client.Options) (client.Client, error) {
  /* source code from pkg/manager/client_builder.go */
  c, err := client.New(config, options)
  if err != nil {
    return nil, err
  }

  return client.NewDelegatingClient(client.NewDelegatingClientInput{
    CacheReader:     cache,
    Client:          c,
    UncachedObjects: n.uncached,
  })
}
```
```golang
func NewDelegatingClient(in NewDelegatingClientInput) (Client, error) {
  /* snipped source code from pkg/client/split.go */
  uncachedGVKs := map[schema.GroupVersionKind]struct{}{}
  /* snipped: collect uncached GVKs associated with in.UncachedObjects */
  
  return &delegatingClient{
    scheme: in.Client.Scheme(),
    mapper: in.Client.RESTMapper(),
    Reader: &delegatingReader{
      CacheReader:  in.CacheReader,
      ClientReader: in.Client,
      scheme:       in.Client.Scheme(),
      uncachedGVKs: uncachedGVKs,
    },
    Writer:       in.Client,
    StatusClient: in.Client,
  }, nil
}
```

All operations that are directly sent to the k8s apiserver rely on client-go's `rest` package under the hood, and all read operations using the cache rely on `SharedIndexInformer`.

## And finally, the Controller
When we create the controller and add it to the manager ([remember?](#Starting-with-the-Manager)), we link it to our CRD type and `Reconciler` implementation. The resulting controller also contains a workqueue constructor which leverages client-go's `workqueue.RateLimitingInterface`.
```golang
func New(name string, mgr manager.Manager, options Options) (Controller, error) {
  /* snipped source code from pkg/controller/controller.go */
  c, err := NewUnmanaged(name, mgr, options)
  if err != nil {
    return nil, err
  }
  return c, mgr.Add(c)
}
func NewUnmanaged(name string, mgr manager.Manager, options Options) (Controller, error) {
  /* snipped source code from pkg/controller/controller.go */
  return &controller.Controller{
      Do: options.Reconciler,
      MakeQueue: func() workqueue.RateLimitingInterface {
        return workqueue.NewNamedRateLimitingQueue(options.RateLimiter, name)
      },
    }, nil
}
```
When the controller is added to the manager, the manager's cache is injected into the controller. When the manager is started, the controller is started as well. This will create and run informers for our resources, and the informers are managed by the injected manager cache. The workqueue is created and the controller invokes the informer's `AddEventHandler` function to register its workqueue to receive resource updates. It then starts worker goroutines, each of which will process a workqueue item by calling our custom reconciler logic inside `processNextWorkItem`.
```golang
func (c *Controller) Start(ctx context.Context) error {
  /* snipped source code from pkg/internal/controller/controller.go */
  // create workqueue
  c.Queue = c.MakeQueue()
  defer c.Queue.ShutDown()

  err := func() error {
    // runs informers, registers workqueue for resource updates
    for _, watch := range c.startWatches {
      c.Log.Info("Starting EventSource", "source", watch.src)
      if err := watch.src.Start(ctx, watch.handler, c.Queue, watch.predicates...); err != nil {
        return err
      }
    }
    // wait until informers are synced
    for _, watch := range c.startWatches {
      syncingSource, ok := watch.src.(source.SyncingSource)
      if !ok {
        continue
      }
    }

    // launch workers to process resources
    for i := 0; i < c.MaxConcurrentReconciles; i++ {
      go wait.UntilWithContext(ctx, func(ctx context.Context) {
        // calls custom reconciler logic
        for c.processNextWorkItem(ctx) {
        }
      }, c.JitterPeriod)
    }

    c.Started = true
    return nil
  }()
  if err != nil {
    return err
  }

  <-ctx.Done()
  // stop workers
  return nil
```

## In a nutshell
And finally, here is a picture to summarize what we have found out:

![dartboard]({{"/images/controller-runtime.svg"}})