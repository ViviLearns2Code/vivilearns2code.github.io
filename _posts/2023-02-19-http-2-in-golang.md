---
layout: post
title:  "HTTP/2 in Golang"
date:   2023-03-11 18:17:42 +0100
categories: misc
comments: true
---
This is the start of a series of posts about two up-and-coming technologies: HTTP/2 and gRPC. Both are closely related and slowly superseding HTTP/1.1.and conventional REST APIs respectively. In this first post, I would like to give a brief introduction to HTTP/2 and how we can enable Golang clients and servers to talk HTTP/2 over both encrypted and unencrypted connections.

## HTTP/2 and its design goals
HTTP/2 was designed as an improvement of HTTP/1.1 mostly with web browsing use cases in mind. In the past, if the browser wanted to load a website and all the numerous resources required to render it (like stylesheets, javascript files, images etc.) from a server, it had to open multiple connections to retrieve these resources in parallel. This is because in HTTP/1.1, one request always takes up an entire TCP connection and blocks it until the server finishes processing and returns the response. This is also called head-of-line blocking. HTTP/2 offers **request multiplexing** as solution: Multiple requests can now be transmitted over a single connection and be processed by the server in an almost parallel manner. This is made possible by splitting requests into little chunks ("frames"). Chunks from different requests can be interleaved when they are sent over the wire, so requests that take less time to process can return without being held up slower requests. Another improvement of HTTP/2 is the introduction of **header compression**. In HTTP/1.1, headers could take on a big size, requiring multiple roundtrips just to be fully transmitted. HTTP/2 introduces a new compression algorithm called hpack, reducing header size drastically. 

These are the main two improvements that I wanted to mention here. HTTP/2 also offers other browser-centric features like server-side push - but to me, request multiplexing and header compression are the two things that are the most interesting from a backend/microservice development perspective. The following loadtest with `h2load`[^1] demonstrates these two key features quite well: Here we run `h2load` against a simple HTTP/2 endpoint which sleeps 10 seconds with `c=1` client (connection) and `n=20` requests. 

```txt
Application protocol: http/1.1
...
finished in 200.60s, 0.10 req/s, 31B/s
requests: 20 total, 20 started, 20 done, 20 succeeded, 0 failed, 0 errored, 0 timeout
status codes: 20 2xx, 0 3xx, 0 4xx, 0 5xx
traffic: 6.09KB (6240) total, 3.91KB (4000) headers (space savings 0.00%), 1.43KB (1460) data
                     min         max         mean         sd        +/- sd
time for request:     10.05s     200.60s     105.32s      59.34s    60.00%
time for connect:      195us       195us       195us         0us   100.00%
time to 1st byte:     10.05s      10.05s      10.05s         0us   100.00%
req/s           :       0.10        0.10        0.10        0.00   100.00%

---------------------------------------------------------
Application protocol: h2c
...
finished in 11.90s, 1.68 req/s, 223B/s
requests: 20 total, 20 started, 20 done, 20 succeeded, 0 failed, 0 errored, 0 timeout
status codes: 20 2xx, 0 3xx, 0 4xx, 0 5xx
traffic: 2.59KB (2655) total, 774B (774) headers (space savings 81.57%), 1.43KB (1460) data
                     min         max         mean         sd        +/- sd
time for request:     10.29s      11.90s      11.38s    443.44ms    75.00%
time for connect:      250us       250us       250us         0us   100.00%
time to 1st byte:     10.29s      10.29s      10.29s         0us   100.00%
req/s           :       1.68        1.68        1.68        0.00   100.00%
```
With HTTP/1.1, all 20 sleep durations are added up to a total 200 seconds while in HTTP/2, all 20 requests just take up roughly 10 seconds. At the same time, we can see a header size reduction of more than 80%.

## Streams, frames and more
Before we proceed to the implementation, let's take a brief look at HTTP/2 specific terminology. In Golang, it is possible to view verbose debug logs for HTTP/2 by setting the environment variable `GODEBUG=http2debug=2`. When a request comes in, the logs look like this:

```log
2023/03/11 18:55:35 reverse proxy server listening at :4343
2023/03/11 18:55:35 grpc server listening at [::]:4242
2023/03/11 18:55:35 http server listening at 4444
2023/03/11 18:55:42 http2: server connection from 127.0.0.1:37178 on 0xc00017c0f0
2023/03/11 18:55:42 http2: Framer 0xc00026c0e0: wrote SETTINGS len=30, settings: MAX_FRAME_SIZE=1048576, MAX_CONCURRENT_STREAMS=250, MAX_HEADER_LIST_SIZE=1048896, HEADER_TABLE_SIZE=4096, INITIAL_WINDOW_SIZE=1048576
2023/03/11 18:55:42 http2: server: client 127.0.0.1:37178 said hello
2023/03/11 18:55:42 http2: Framer 0xc00026c0e0: read SETTINGS len=18, settings: ENABLE_PUSH=0, INITIAL_WINDOW_SIZE=4194304, MAX_HEADER_LIST_SIZE=10485760
2023/03/11 18:55:42 http2: server read frame SETTINGS len=18, settings: ENABLE_PUSH=0, INITIAL_WINDOW_SIZE=4194304, MAX_HEADER_LIST_SIZE=10485760
2023/03/11 18:55:42 http2: server processing setting [ENABLE_PUSH = 0]
2023/03/11 18:55:42 http2: server processing setting [INITIAL_WINDOW_SIZE = 4194304]
2023/03/11 18:55:42 http2: server processing setting [MAX_HEADER_LIST_SIZE = 10485760]
2023/03/11 18:55:42 http2: Framer 0xc00026c0e0: read WINDOW_UPDATE len=4 (conn) incr=1073741824
2023/03/11 18:55:42 http2: server read frame WINDOW_UPDATE len=4 (conn) incr=1073741824
2023/03/11 18:55:42 http2: Framer 0xc00026c0e0: read HEADERS flags=END_STREAM|END_HEADERS stream=1 len=62
2023/03/11 18:55:42 http2: decoded hpack field header field ":authority" = "localhost:4646"
2023/03/11 18:55:42 http2: decoded hpack field header field ":method" = "GET"
2023/03/11 18:55:42 http2: decoded hpack field header field ":path" = "/hello"
2023/03/11 18:55:42 http2: decoded hpack field header field ":scheme" = "https"
2023/03/11 18:55:42 http2: decoded hpack field header field "accept-encoding" = "gzip"
2023/03/11 18:55:42 http2: decoded hpack field header field "user-agent" = "Go-http-client/2.0"
2023/03/11 18:55:42 http2: decoded hpack field header field "x-forwarded-for" = "127.0.0.1"
2023/03/11 18:55:42 http2: server read frame HEADERS flags=END_STREAM|END_HEADERS stream=1 len=62
2023/03/11 18:55:42 http2: Framer 0xc00026c0e0: wrote SETTINGS flags=ACK len=0
2023/03/11 18:55:42 http2: Framer 0xc00026c0e0: wrote WINDOW_UPDATE len=4 (conn) incr=983041
2023/03/11 18:55:42 http2: Framer 0xc00026c0e0: read SETTINGS flags=ACK len=0
2023/03/11 18:55:42 proxy directing request to http://localhost:4444/hello
2023/03/11 18:55:42 http2: server read frame SETTINGS flags=ACK len=0
2023/03/11 18:55:42 http2: Transport creating client conn 0xc0000b2300 to 127.0.0.1:4444
2023/03/11 18:55:42 http2: Framer 0xc0000cc0e0: wrote SETTINGS len=18, settings: ENABLE_PUSH=0, INITIAL_WINDOW_SIZE=4194304, MAX_HEADER_LIST_SIZE=10485760
2023/03/11 18:55:42 http2: Framer 0xc0000cc0e0: wrote WINDOW_UPDATE len=4 (conn) incr=1073741824
2023/03/11 18:55:42 http2: Transport encoding header ":authority" = "localhost:4646"
2023/03/11 18:55:42 http2: Transport encoding header ":method" = "GET"
2023/03/11 18:55:42 http2: Transport encoding header ":path" = "/hello"
2023/03/11 18:55:42 http2: Transport encoding header ":scheme" = "http"
2023/03/11 18:55:42 http2: Transport encoding header "accept-encoding" = "gzip"
2023/03/11 18:55:42 http2: Transport encoding header "user-agent" = "Go-http-client/2.0"
2023/03/11 18:55:42 http2: Transport encoding header "x-forwarded-for" = "127.0.0.1, 127.0.0.1"
2023/03/11 18:55:42 http2: Framer 0xc0000cc0e0: wrote HEADERS flags=END_STREAM|END_HEADERS stream=3 len=69
2023/03/11 18:55:42 h2c: attempting h2c with prior knowledge.
2023/03/11 18:55:42 http2: server connection from 127.0.0.1:59270 on 0xc00017c2d0
2023/03/11 18:55:42 http2: Framer 0xc0000cc460: wrote SETTINGS len=30, settings: MAX_FRAME_SIZE=1048576, MAX_CONCURRENT_STREAMS=250, MAX_HEADER_LIST_SIZE=1048896, HEADER_TABLE_SIZE=4096, INITIAL_WINDOW_SIZE=1048576
2023/03/11 18:55:42 http2: Framer 0xc0000cc460: read SETTINGS len=18, settings: ENABLE_PUSH=0, INITIAL_WINDOW_SIZE=4194304, MAX_HEADER_LIST_SIZE=10485760
2023/03/11 18:55:42 http2: Framer 0xc0000cc460: wrote WINDOW_UPDATE len=4 (conn) incr=983041
2023/03/11 18:55:42 http2: server read frame SETTINGS len=18, settings: ENABLE_PUSH=0, INITIAL_WINDOW_SIZE=4194304, MAX_HEADER_LIST_SIZE=10485760
2023/03/11 18:55:42 http2: server processing setting [ENABLE_PUSH = 0]
2023/03/11 18:55:42 http2: server processing setting [INITIAL_WINDOW_SIZE = 4194304]
2023/03/11 18:55:42 http2: server processing setting [MAX_HEADER_LIST_SIZE = 10485760]
2023/03/11 18:55:42 http2: Framer 0xc0000cc460: read WINDOW_UPDATE len=4 (conn) incr=1073741824
2023/03/11 18:55:42 http2: server read frame WINDOW_UPDATE len=4 (conn) incr=1073741824
2023/03/11 18:55:42 http2: Framer 0xc0000cc460: read HEADERS flags=END_STREAM|END_HEADERS stream=3 len=69
2023/03/11 18:55:42 http2: decoded hpack field header field ":authority" = "localhost:4646"
2023/03/11 18:55:42 http2: decoded hpack field header field ":method" = "GET"
2023/03/11 18:55:42 http2: decoded hpack field header field ":path" = "/hello"
2023/03/11 18:55:42 http2: decoded hpack field header field ":scheme" = "http"
2023/03/11 18:55:42 http2: decoded hpack field header field "accept-encoding" = "gzip"
2023/03/11 18:55:42 http2: decoded hpack field header field "user-agent" = "Go-http-client/2.0"
2023/03/11 18:55:42 http2: decoded hpack field header field "x-forwarded-for" = "127.0.0.1, 127.0.0.1"
2023/03/11 18:55:42 http2: server read frame HEADERS flags=END_STREAM|END_HEADERS stream=3 len=69
2023/03/11 18:55:42 http2: Framer 0xc0000cc0e0: read SETTINGS len=30, settings: MAX_FRAME_SIZE=1048576, MAX_CONCURRENT_STREAMS=250, MAX_HEADER_LIST_SIZE=1048896, HEADER_TABLE_SIZE=4096, INITIAL_WINDOW_SIZE=1048576
2023/03/11 18:55:42 http2: Framer 0xc0000cc460: wrote SETTINGS flags=ACK len=0
2023/03/11 18:55:42 http2: Transport received SETTINGS len=30, settings: MAX_FRAME_SIZE=1048576, MAX_CONCURRENT_STREAMS=250, MAX_HEADER_LIST_SIZE=1048896, HEADER_TABLE_SIZE=4096, INITIAL_WINDOW_SIZE=1048576
2023/03/11 18:55:42 http2: Framer 0xc0000cc0e0: wrote SETTINGS flags=ACK len=0
2023/03/11 18:55:42 http2: Framer 0xc0000cc0e0: read WINDOW_UPDATE len=4 (conn) incr=983041
2023/03/11 18:55:42 http2: Transport received WINDOW_UPDATE len=4 (conn) incr=983041
2023/03/11 18:55:42 http2: Framer 0xc0000cc460: read SETTINGS flags=ACK len=0
2023/03/11 18:55:42 http2: server read frame SETTINGS flags=ACK len=0
2023/03/11 18:55:42 http2: Framer 0xc0000cc0e0: read SETTINGS flags=ACK len=0
2023/03/11 18:55:42 http2: Transport received SETTINGS flags=ACK len=0
2023/03/11 18:55:42 http2: server encoding header ":status" = "200"
2023/03/11 18:55:42 http2: server encoding header "content-type" = "text/plain; charset=utf-8"
2023/03/11 18:55:42 http2: server encoding header "content-length" = "43"
2023/03/11 18:55:42 http2: server encoding header "date" = "Sat, 11 Mar 2023 17:55:42 GMT"
2023/03/11 18:55:42 http2: Framer 0xc0000cc460: wrote HEADERS flags=END_HEADERS stream=3 len=49
2023/03/11 18:55:42 http2: Framer 0xc0000cc460: wrote DATA flags=END_STREAM stream=3 len=43 data="Hello, /hello, TLS: false, Proto: HTTP/2.0\n"
2023/03/11 18:55:42 http2: Framer 0xc0000cc0e0: read HEADERS flags=END_HEADERS stream=3 len=49
2023/03/11 18:55:42 http2: decoded hpack field header field ":status" = "200"
2023/03/11 18:55:42 http2: decoded hpack field header field "content-type" = "text/plain; charset=utf-8"
2023/03/11 18:55:42 http2: decoded hpack field header field "content-length" = "43"
2023/03/11 18:55:42 http2: decoded hpack field header field "date" = "Sat, 11 Mar 2023 17:55:42 GMT"
2023/03/11 18:55:42 http2: Transport received HEADERS flags=END_HEADERS stream=3 len=49
2023/03/11 18:55:42 http2: Framer 0xc0000cc0e0: read DATA flags=END_STREAM stream=3 len=43 data="Hello, /hello, TLS: false, Proto: HTTP/2.0\n"
2023/03/11 18:55:42 http2: Transport received DATA flags=END_STREAM stream=3 len=43 data="Hello, /hello, TLS: false, Proto: HTTP/2.0\n"
2023/03/11 18:55:42 http2: server encoding header ":status" = "200"
2023/03/11 18:55:42 http2: server encoding header "content-type" = "text/plain; charset=utf-8"
2023/03/11 18:55:42 http2: server encoding header "date" = "Sat, 11 Mar 2023 17:55:42 GMT"
2023/03/11 18:55:42 http2: server encoding header "content-length" = "43"
2023/03/11 18:55:42 http2: Framer 0xc00026c0e0: wrote HEADERS flags=END_HEADERS stream=1 len=49
2023/03/11 18:55:42 http2: Framer 0xc00026c0e0: wrote DATA flags=END_STREAM stream=1 len=43 data="Hello, /hello, TLS: false, Proto: HTTP/2.0\n"+
```

This is a lot of information, and a lot of it has to do with flow control. Flow control ensures that the client will only send as much data over the connection at a time as is possible for the server to process. In the above logs, this is what client and server do using `SETTINGS` and `WINDOW_UPDATE` frames. What is a `frame`? A frame is _"the smallest unit of communication in HTTP/2, each belonging to a stream"_ [^2]. This brings us to streams: a `stream` is a _"bidirectional flow of bytes within an established connection"_ [^2]. In the logs, we can for example see the response `Hello, /hello, ... ` being sent as part of the stream with id `stream=3`. We can also see the new HTTP/2-specific pseudo-header fields `:authority`, `:method`, `:path`, `:scheme` which now contain the request metadata. These pseudo-fields are not accessible by the programmer in Golang and their usage definition often stricter [^3]. In addition, we see the logs for header encoding and decoding using hpack.

## End-to-end HTTP/2 client/server scenario in Golang
To demonstrate HTTP/2 over both encrypted and unencrypted connections, we will implement the following scenario:

![h2 setup]({{"/images/h2Setup.png"}})

In this scenario, we have two proxies involved between client and server. The client proxy will receive the client's outgoing request in cleartext and forward it to the server proxy over a TLS encrypted connection. The server proxy will then forward the request over cleartext to the server. In total, we have 3 hops: client -> proxy, proxy -> proxy, proxy -> server. We want each of these hops to use HTTP/2. gRPC for instance only works if we can guarantee end-to-end HTTP/2.

The repo with the implementation can be found here: https://github.com/ViviLearns2Code/h2-and-grpc

For the rest of this post, I will only be referring to snippets from the repo.

In Golang, both proxies can be easily implemented using `httputil.ReverseProxy` handlers[^4].

The easiest hop to implement is the one that uses TLS encryption. TLS comes with a functionality called ALPN (Application-Layer Protocol Negotiation) that allows client and server to negotiate the protocol at run time during the TLS handshake without knowing what the other supports at design time. Each part simply specifies its protocol preference in its `tls.Config.NextProtos` field. Note that for the client transport, it is possible to use both `http.Transport` and `http2.Transport`. However, the former needs to be additionally configured with `http2.ConfigureTransport`.

```golang
// client
// https://github.com/ViviLearns2Code/h2-and-grpc/blob/main/client/main.go#L87
    // ...snip
	// configured http.Transport interchangeable with http2.Transport for h2 to work
	proxyTrans := &http.Transport{
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: true,
			NextProtos:         []string{"http/1.1", "h2"},
		},
	}
	if err := http2.ConfigureTransport(proxyTrans); err != nil {
		log.Print(err.Error())
	}
	proxy := httputil.ReverseProxy{
		Director:  dirFn,
		Transport: proxyTrans,
	}
	revProx := &http.Server{
		Addr:    fmt.Sprintf(":%d", proxyPort),
		Handler: h2c.NewHandler(&proxy, &http2.Server{}),
	}
    // snip...

// server
// https://github.com/ViviLearns2Code/h2-and-grpc/blob/main/server/main.go#L105
    // ...snip
	revProx := &http.Server{
		Addr:    fmt.Sprintf(":%d", port),
		Handler: router,
		TLSConfig: &tls.Config{
			NextProtos: []string{"h2", "http/1.1"},
			ClientAuth: tls.NoClientCert,
		},
	}
    // snip...
```
During ALPN, the first protocol from the server's list which the client speaks (but not necessarily prefers) will be the chosen protocol. In other words, server preference takes precedence. In this example, both will negotiate to use `h2`.

Next, let's look at the HTTP/2 cleartext connection between the client app and the client proxy (this is also termed `h2c` - HTTP/2 over cleartext). First of all, the client app needs to determine the transport type at design time, as we no longer have ALPN to determine the protocol dynamically. The client app can no longer use `http.Transport` but needs to encode the knowledge that the server can speak HTTP/2 by using `http2.Transport`. Such requests are said to be sent with prior knowledge [^5]. In Golang however, merely using the right transport type is not enough: By design, Golang's standard library requires TLS for HTTP/2 to work. In order to skip TLS, We need to apply a hack and overwrite the transport's `DialTLSContext` function:
```golang
// client
// https://github.com/ViviLearns2Code/h2-and-grpc/blob/main/client/main.go#L40
    // ...snip
	client := &http.Client{}
	t2 := &http2.Transport{
		AllowHTTP: true,
		DialTLSContext: func(ctx context.Context, network, addr string, cfg *tls.Config) (net.Conn, error) {
			// fake TLS to allow HTTP
			var d net.Dialer
			return d.DialContext(ctx, network, addr)
		},
	}
	client.Transport = t2

	resp, err := client.Get(addr)
    // snip...
```
On the other side of the connection, the client proxy handler needs to be wrapped with `h2c.NewHandler` to support HTTP/2 over cleartext. The wrapper will simply intercept the request, check if it is HTTP/2 and if yes, it will hand it over to the standard library's h2 logic, otherwise it will directly hand it over to our handler.
```golang
// client
// https://github.com/ViviLearns2Code/h2-and-grpc/blob/main/client/main.go#L102
    // ...snip
	revProx := &http.Server{
		Addr:    fmt.Sprintf(":%d", proxyPort),
		Handler: h2c.NewHandler(&proxy, &http2.Server{}),
	}
    // snip...
```
With this, we have successfully enabled both client app and client proxy to send and receive HTTP/2 requests over cleartext. The same kind of adjustments can be made for the cleartext connection between server proxy and server app.


## Next up: gRPC
In this post, I have introduced the basic ideas behind HTTP/2 and how we can implement a simple scenario using the new protocol version over both encrypted and unencrypted connections. In my next post, I will dive into gRPC - its promises, quirks and how to use it for your own projects.

[^1]: `h2load`  offers support for both protocol versions and supersedes benchmarking tools like `ab` which only support HTTP/1.x
[^2]: https://web.dev/performance-http2/
[^3]: https://stackoverflow.com/a/70512226
[^4]: https://pkg.go.dev/net/http/httputil#ReverseProxy
[^5]: https://httpwg.org/specs/rfc7540.html#known-http
