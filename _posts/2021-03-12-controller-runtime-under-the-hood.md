---
layout: post
title:  "Controller Runtime Under The Hood"
date:   2021-03-12 17:16:30 +0200
categories: k8s
comments: true
---
When I first got started with controllers, it was difficult for me to understand how `client-go`, `controller-runtime` and `kubebuilder` were related to each other. I read up on [client-go concepts](https://github.com/kubernetes/sample-controller/blob/master/docs/controller-client-go.md), and knew that kubebuilder was based on controller-runtime and controller-runtime somehow based on client-go, but it wasn't clear to me where for example client-go's informers came into controller-runtime. This post will hopefully help other curious people like me to get a better picture of controller-runtime.

We start our exploration of controller-runtime from our simplified custom code, so that it is easier to see what we need to write ourselves and where the library comes in. Basically, we need to do the following things ourselves:
* define and register CRD types
* implement `Reconciler` interface
* generate deepcopy functions
* create and start controller-runtime manager and controller

```golang
/* snipped CRD golang type definitions in our custom code myoperator/api/v1 */
package v1
import (
  "k8s.io/apimachinery/pkg/runtime/schema"
  "sigs.k8s.io/controller-runtime/pkg/scheme"
)

var (
  // GroupVersion is group version used to register these objects
  GroupVersion = schema.GroupVersion{Group: "mygroup.mydomain", Version: "v1"}
  // SchemeBuilder is used to add go types to the GroupVersionKind scheme
  SchemeBuilder = &scheme.Builder{GroupVersion: GroupVersion}
  // AddToScheme adds the types in this group-version to the given scheme.
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
/* source code snipped from controller-runtime's pkg/reconcile/reconcile.go */
type Reconciler interface {
  Reconcile(context.Context, Request) (Result, error)
}
```
```golang
/* snipped CRD controller implementation in our custom code myoperator/controllers */
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
If you do not use kubebuilder, you need to generate a file with deepcopy functions for your resource manually, to satisfy the `runtime.Object` interface. This can be done using [code-generator](https://github.com/kubernetes/code-generator). It offers four generators, but we only need to run one `deecopy-gen`. The other generators `client-gen`, `informer-gen`, `lister-gen` are not required because controller-runtime will handle the clientset, informer and lister objects for us. For more information on how to use `deepcopy-gen`, checkout this [article](https://www.openshift.com/blog/kubernetes-deep-dive-code-generation-customresources).
```golang
/* snipped, auto-generated source code from api/v1/zz_generated_deepcopy.go */
package v1

import (
  "k8s.io/apimachinery/pkg/runtime"
)

func (in *MyKind) DeepCopyInto(out *MyKind) { /* snipped */ }
func (in *MyKind) DeepCopy() *MyKind { /* snipped */ }
func (in *MyKind) DeepCopyObject() runtime.Object { /* snipped */ }
func (in *MyKindList) DeepCopyInto(out *MyKindList) { /* snipped */ }
func (in *MyKindList) DeepCopy() *MyKindList { /* snipped */ }
func (in *MyKindList) DeepCopyObject() runtime.Object { /* snipped */ }
func (in *MyKindSpec) DeepCopyInto(out *MyKindSpec) { /* snipped */ }
func (in *MyKindSpec) DeepCopy() *MyKindSpec { /* snipped */ }
func (in *MyKindStatus) DeepCopyInto(out *MyKindStatus) { /* snipped */ }
func (in *MyKindStatus) DeepCopy() *MyKindStatus { /* snipped */ }
```

Now that we have our custom logic ready, it is time to use controller-runtime.

## Starting with the Manager
A manager is required to create and start up a controller. The manager needs a kubeconfig (controller-runtime offers `GetConfigOrDie()` to read it from the system) and can be provided with many options. Two examples are `Scheme`, which contains predefined k8s types and our CRD type, or the `Port` used to serve webhooks on. After creating and configuring the manager, we can add our CRD controller to it and start the manager.
```golang
/* snipped source code from our application main.go */
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
This is just a high-level overview of what you would need to do. But what happens under the hood? How is the manager related to client-go concepts such as Informers?

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
|`Start`| runs all informers, meaning it will list & watch the k8s apiserver for resource updates and add objects to the Delta Fifo queue | `cache.ListWatch` |
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

![img](images/controller-runtime.svg)