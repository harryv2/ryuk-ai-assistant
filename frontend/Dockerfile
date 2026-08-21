# The UI image. `docker compose up` mounts the source over /app and runs the
# dev server, so the default target is `dev`. A static build is one flag away:
#
#   docker build --target prod -t alpha-law-ui ./frontend
#
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json ./
RUN npm install --no-audit --no-fund

FROM node:22-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:22-alpine AS prod
WORKDIR /app
ENV NODE_ENV=production
COPY --from=deps /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY package.json vite.config.ts ./
EXPOSE 5173
CMD ["npx", "vite", "preview", "--host", "0.0.0.0", "--port", "5173"]

FROM node:22-alpine AS dev
WORKDIR /app
ENV NODE_ENV=development
# Inside the compose network the API answers on the service name, not localhost.
ENV VITE_API_TARGET=http://api:8000
ENV CHOKIDAR_USEPOLLING=1
COPY --from=deps /app/node_modules ./node_modules
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
