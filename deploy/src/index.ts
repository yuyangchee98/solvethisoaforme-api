import { Container, getContainer } from "@cloudflare/containers";

interface Env {
	API_CONTAINER: DurableObjectNamespace<ApiContainer>;
	JWT_SECRET: string;
	GOOGLE_VISION_API_KEY: string;
	FRONTEND_URL: string;
	// Optional: set for self-host mode; leave unset for pure BYOK.
	ANTHROPIC_API_KEY?: string;
}

export class ApiContainer extends Container<Env> {
	defaultPort = 8000;
	// Phase 1 runs without durable persistence — everything on disk is lost
	// when the container sleeps, so keep the idle window generous.
	sleepAfter = "2h";
	envVars = {
		JWT_SECRET: this.env.JWT_SECRET,
		GOOGLE_VISION_API_KEY: this.env.GOOGLE_VISION_API_KEY,
		FRONTEND_URL: this.env.FRONTEND_URL,
		DATA_PATH: "/data",
		...(this.env.ANTHROPIC_API_KEY
			? { ANTHROPIC_API_KEY: this.env.ANTHROPIC_API_KEY }
			: {}),
	};

	override async fetch(request: Request): Promise<Response> {
		// The app runs DB migrations before uvicorn binds its port, which can
		// exceed the library's default 20s port wait (TIMEOUT_TO_GET_PORTS_MS)
		// on a cold start — and a failed wait leaves the DO wedged. Wait
		// generously here; once the container is up this resolves instantly.
		await this.startAndWaitForPorts({
			ports: this.defaultPort,
			cancellationOptions: {
				portReadyTimeoutMS: 180_000,
				abort: request.signal,
			},
		});
		return super.fetch(request);
	}
}

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		// Single fixed instance: the app assumes one process (SQLite file,
		// in-memory session workers), so every request routes to "main".
		return getContainer(env.API_CONTAINER, "main").fetch(request);
	},
} satisfies ExportedHandler<Env>;
